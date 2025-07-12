#  Copyright 2025 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import time
from typing import Any, Generator
from unittest.mock import Mock, patch

import pytest
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from app.amqp.amqp_handler import ProcessAmqpRequestHandler
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.processing import ProcessingItem


@pytest.fixture
def app_config() -> ApplicationConfig:
    """Create test ApplicationConfig"""
    return ApplicationConfig(
        esHost="localhost:9200",
        logLevel="DEBUG",
        amqpUrl="amqp://guest:guest@localhost:5672/",
        amqpExchangeName="test_analyzer",
        appVersion="test",
        instanceTaskType="test",
        amqpHandlerTaskTimeout=6,
    )


@pytest.fixture
def search_config() -> SearchConfig:
    """Create test SearchConfig"""
    return SearchConfig()


@pytest.fixture
def mock_amqp_client() -> Mock:
    """Create mocked AmqpClient"""
    client = Mock()
    client.reply = Mock()
    return client


@pytest.fixture
def handler(app_config, search_config, mock_amqp_client) -> Generator[ProcessAmqpRequestHandler, Any, None]:
    """Create ProcessAmqpRequestHandler instance with mocked client"""
    handler = ProcessAmqpRequestHandler(
        app_config=app_config,
        search_config=search_config,
        queue_size=10,
        client=mock_amqp_client,
        init_services=["noop_echo", "noop_sleep", "noop_fail"],
    )
    yield handler
    # Clean up
    handler.shutdown()


def create_test_processing_item(
    routing_key="noop_echo", item="test_data", reply_to="test_reply", priority=1000
) -> ProcessingItem:
    """Helper to create ProcessingItem for testing"""
    return ProcessingItem(
        priority=priority,
        number=1,
        routing_key=routing_key,
        reply_to=reply_to,
        log_correlation_id="test_correlation",
        msg_correlation_id="test_msg_correlation",
        item=item,
    )


def create_amqp_request_mock(
    routing_key="noop_echo", body: Any = "test_data", reply_to="test_reply", correlation_id="test_correlation"
):
    """Helper to create AMQP request mocks"""
    channel = Mock(spec=BlockingChannel)
    channel.basic_ack = Mock()

    method = Mock(spec=Basic.Deliver)
    method.routing_key = routing_key
    method.delivery_tag = 123

    props = Mock(spec=BasicProperties)
    props.reply_to = reply_to
    props.correlation_id = correlation_id
    props.headers = None

    body_bytes = json.dumps(body).encode("utf-8")

    return channel, method, props, body_bytes


def custom_retry_predicate(item, _):
    """Custom retry predicate that only retries tasks with 'should_retry' in the item data"""
    return "should_retry" in str(item.item)


class TestProcessAmqpRequestHandler:
    """Test suite for ProcessAmqpRequestHandler"""

    def test_task_processing_lifecycle(self, handler, mock_amqp_client):
        """Test Case 1: Processing a task - check it appears in running_tasks and then disappears"""
        # Create a task with short sleep to monitor lifecycle
        channel, method, props, body = create_amqp_request_mock(
            routing_key="noop_sleep", body=1, reply_to="test_reply"  # Sleep for 1 seconds
        )

        # Verify initially no running tasks
        assert len(handler.running_tasks) == 0

        # Submit task
        handler.handle_amqp_request(channel, method, props, body)

        # Wait a bit for task to be picked up and sent to processor
        time.sleep(0.1)

        # Verify task appears in running_tasks
        running_tasks = handler.running_tasks
        assert len(running_tasks) == 1, "Task should appear in running_tasks"
        running_task = running_tasks[0]
        assert running_task.routing_key == "noop_sleep"
        assert running_task.item == 1
        assert running_task.send_time is not None

        # Wait for task to complete (noop_sleep sleeps for 0.2s + processing time)
        time.sleep(11)

        # Verify task disappears from running_tasks
        assert len(handler.running_tasks) == 0, "Task should be removed from running_tasks after completion"

        # Verify basic_ack was called on the channel
        channel.basic_ack.assert_called_once_with(delivery_tag=123)

    def test_getting_response(self, handler, mock_amqp_client):
        """Test Case 2: Send a task with test string, check that it was returned to mocked AmqpClient"""
        test_message = "Hello, World!"

        # Create task with noop_echo handler (returns input as-is)
        channel, method, props, body = create_amqp_request_mock(
            routing_key="noop_echo",
            body=test_message,
            reply_to="test_reply_queue",
            correlation_id="test_correlation_123",
        )

        # Submit the task
        handler.handle_amqp_request(channel, method, props, body)

        # Wait for processing to complete
        time.sleep(10)

        # Verify that reply was called on the mock client with correct parameters
        mock_amqp_client.reply.assert_called_once()
        call_args = mock_amqp_client.reply.call_args

        # Check reply parameters
        assert call_args[0][0] == "test_reply_queue", "Reply should be sent to correct queue"
        assert call_args[0][1] == "test_correlation_123", "Correlation ID should match"
        assert call_args[0][2] == test_message, "Response body should match the echoed input"

    # noinspection PyUnresolvedReferences
    def test_killing_processing_process(self, handler, mock_amqp_client):
        """Test Case 3: Kill processing process, check restart and task requeue"""
        # Create a longer-running task
        channel, method, props, body = create_amqp_request_mock(
            routing_key="noop_sleep", body=2, reply_to="test_reply"  # Sleep for 2 seconds
        )

        # Submit the long-running task
        handler.handle_amqp_request(channel, method, props, body)

        # Wait for task to be picked up
        time.sleep(0.1)

        # Verify task is in running_tasks
        assert len(handler.running_tasks) == 1
        original_task = list(handler.__running_tasks.queue)[0]
        original_send_time = original_task.send_time
        original_process_pid = handler.processor.pid

        # Kill the processor process
        handler.processor.process.terminate()
        handler.processor.process.join(timeout=1)

        # Wait for the handler to detect process death and restart
        time.sleep(0.5)

        # Verify process was restarted (new PID)
        assert handler.processor.is_alive(), "Process should be restarted"
        new_process_pid = handler.processor.pid
        assert new_process_pid != original_process_pid, "New process should have different PID"

        # Verify task was requeued with new send time
        assert len(handler.running_tasks) == 1, "Task should still be in running_tasks"
        requeued_task = list(handler.__running_tasks.queue)[0]
        assert requeued_task.routing_key == original_task.routing_key
        assert requeued_task.item == original_task.item
        assert requeued_task.send_time > original_send_time, "Task should have new send_time after restart"

        # Wait for the requeued task to complete
        time.sleep(10)

        # Verify task eventually completes
        assert len(handler.running_tasks) == 0, "Requeued task should eventually complete"

    def test_multiple_tasks_processing(self, handler, mock_amqp_client):
        """Test multiple tasks are processed correctly"""
        tasks_data = ["task1", "task2", "task3"]

        # Submit multiple tasks
        for i, task_data in enumerate(tasks_data):
            channel, method, props, body = create_amqp_request_mock(
                routing_key="noop_echo", body=task_data, reply_to=f"reply_queue_{i}", correlation_id=f"correlation_{i}"
            )
            handler.handle_amqp_request(channel, method, props, body)

        # Wait for all tasks to complete
        time.sleep(10)

        # Verify all replies were sent
        assert mock_amqp_client.reply.call_count == len(tasks_data)

        # Verify no tasks remain in running_tasks
        assert len(handler.running_tasks) == 0

    def test_priority_queue_ordering(self, handler, mock_amqp_client):
        """Test that priority queue processes higher priority tasks first"""
        # Submit tasks with different priorities (lower number = higher priority)
        high_priority_task = create_test_processing_item(routing_key="noop_echo", item="high_priority", priority=100)
        low_priority_task = create_test_processing_item(routing_key="noop_echo", item="low_priority", priority=1000)

        # Add tasks directly to queue (simulating handle_amqp_request behavior)
        handler.queue.put(low_priority_task)  # Add low priority first
        handler.queue.put(high_priority_task)  # Add high priority second

        # Wait for processing
        time.sleep(10)

        # Verify both tasks were processed
        assert mock_amqp_client.reply.call_count == 2

        # The first reply should be for the high priority task
        first_call = mock_amqp_client.reply.call_args_list[0]
        assert first_call[0][2] == "high_priority", "Higher priority task should be processed first"

    def test_error_handling_malformed_json(self, handler, mock_amqp_client):
        """Test handling of malformed JSON in message body"""
        channel = Mock(spec=BlockingChannel)
        channel.basic_ack = Mock()
        channel.basic_nack = Mock()

        method = Mock(spec=Basic.Deliver)
        method.routing_key = "noop_echo"
        method.delivery_tag = 123

        props = Mock(spec=BasicProperties)
        props.reply_to = "test_reply"
        props.correlation_id = "test_correlation"
        props.headers = None

        # Malformed JSON body
        malformed_body = b'{"invalid": json data}'

        # Submit malformed request
        handler.handle_amqp_request(channel, method, props, malformed_body)

        # Wait a bit
        time.sleep(0.1)

        # Verify message was nacked due to JSON parsing error
        channel.basic_nack.assert_called_once_with(delivery_tag=123, requeue=False)

        # Verify no reply was sent for malformed message
        mock_amqp_client.reply.assert_not_called()

    # noinspection PyUnreachableCode,PyTestUnpassedFixture
    def test_shutdown_cleanup(self, app_config, search_config, mock_amqp_client):
        """Test proper cleanup during shutdown"""
        handler = ProcessAmqpRequestHandler(
            app_config=app_config,
            search_config=search_config,
            client=mock_amqp_client,
            init_services=["noop_echo"],
        )

        try:
            # Verify handler is initialized
            assert handler._monitor_thread is not None
            assert handler._monitor_thread.is_alive()
            assert handler._receiver_thread is not None
            assert handler._receiver_thread.is_alive()
            assert handler._sender_thread is not None
            assert handler._sender_thread.is_alive()
            assert handler.processor.is_alive()
        finally:
            # Ensure shutdown is called even if test fails
            handler.shutdown()

        # Wait for cleanup
        time.sleep(0.2)

        # Verify cleanup
        assert handler._shutdown is True
        assert not handler.processor.is_alive()
        assert not handler._monitor_thread.is_alive()
        assert not handler._receiver_thread.is_alive()
        assert not handler._sender_thread.is_alive()

    def test_routing_key_predicate_filtering(self, app_config, search_config, mock_amqp_client):
        """Test routing key predicate filtering"""

        # Create handler with routing key predicate that filters out "filtered_key"
        def routing_key_predicate(key):
            return key != "filtered_key"

        handler = ProcessAmqpRequestHandler(
            app_config=app_config,
            search_config=search_config,
            client=mock_amqp_client,
            routing_key_predicate=routing_key_predicate,
            init_services=["noop_echo"],
        )

        try:
            # Submit task with filtered routing key
            channel, method, props, body = create_amqp_request_mock(routing_key="filtered_key", body="test_data")

            handler.handle_amqp_request(channel, method, props, body)

            # Wait a bit
            time.sleep(0.2)

            # Verify task was filtered out (not processed)
            assert len(handler.running_tasks) == 0
            mock_amqp_client.reply.assert_not_called()
        finally:
            handler.shutdown()

    def test_long_running_task_interruption(self, handler, mock_amqp_client):
        """Test Case 4: Long running task interruption - verify task restart when timeout is exceeded"""

        task_timeout = 7
        # Create a task that will run for 7 seconds (longer than 6-second timeout)
        channel, method, props, body = create_amqp_request_mock(
            routing_key="noop_sleep", body=task_timeout, reply_to="test_reply"  # Sleep for 6 seconds
        )

        # Submit the long-running task
        handler.handle_amqp_request(channel, method, props, body)

        # Wait for task to be picked up and record original start time
        time.sleep(0.1)

        # Verify task is in running_tasks and record start time
        running_tasks = handler.running_tasks
        assert len(running_tasks) == 1, "Task should appear in running_tasks"
        original_task = running_tasks[0]
        original_send_time = original_task.send_time
        assert original_send_time is not None, "Task should have send_time recorded"
        assert original_task.routing_key == "noop_sleep"
        assert original_task.item == task_timeout

        # Wait 10 seconds for timeout detection and restart
        # This should be enough for the handler to detect the long-running task (after 5 seconds)
        # and restart the processor
        time.sleep(12)

        # Verify task is still running but was restarted with newer start time
        running_tasks = handler.running_tasks
        assert len(running_tasks) == 1, "Task should still be in running_tasks after restart"
        restarted_task = running_tasks[0]
        assert restarted_task.routing_key == original_task.routing_key
        assert restarted_task.item == original_task.item
        assert restarted_task.send_time > original_send_time, "Task should have newer send_time after restart"
        assert restarted_task.retries == 1, "Task should have incremented retries after restart"

        # Verify processor was restarted (process should be alive)
        assert handler.processor.is_alive(), "Processor should be alive after restart"

    @patch("app.amqp.amqp_handler.logger")
    def test_exception_retry(self, mock_logger, handler, mock_amqp_client):
        """Test Case 5: Failed tasks should be processed several times before dropping"""

        # Create a task that will fail with noop_fail handler
        channel, method, props, body = create_amqp_request_mock(
            routing_key="noop_fail", body="exception_retry", reply_to="test_reply"
        )

        # Submit the failing task
        handler.handle_amqp_request(channel, method, props, body)

        # Wait for task to be picked up and record original start time
        time.sleep(0.01)

        # Verify task is in running_tasks and record start time
        running_tasks = handler.running_tasks
        assert len(running_tasks) == 1, "Task should appear in running_tasks"
        original_task = running_tasks[0]
        original_send_time = original_task.send_time
        assert original_send_time is not None, "Task should have send_time recorded"
        assert original_task.routing_key == "noop_fail"
        assert original_task.item == "exception_retry"

        # Wait for all tasks to complete
        time.sleep(15)

        # Verify task was dropped after retries
        assert len(handler.running_tasks) == 0, "Task should be dropped after retries"

        mock_error = mock_logger.error
        # Verify error was called for failed retries
        error_calls = [call for call in mock_error.call_args_list if "failed after 2 retries." in str(call)]
        assert len(error_calls) == 1, "Should log error message about failed retries"

        mock_amqp_client.reply.assert_not_called()
        assert handler.processor.is_alive(), "Processor should be alive after handling failed task"

    @patch("app.amqp.amqp_handler.logger")
    def test_custom_retry_predicate(self, mock_logger, app_config, search_config, mock_amqp_client):
        """Test Case 7: Custom retry predicate - verify that custom retry logic is respected"""

        # Create handler with custom retry predicate
        handler = ProcessAmqpRequestHandler(
            app_config=app_config,
            search_config=search_config,
            client=mock_amqp_client,
            retry_predicate=custom_retry_predicate,
            init_services=["noop_fail"],
        )

        try:
            # Test Case 7a: Task that should be retried according to custom predicate
            channel_retry, method_retry, props_retry, body_retry = create_amqp_request_mock(
                routing_key="noop_fail",
                body="should_retry_task",
                reply_to="test_reply_retry",
                correlation_id="correlation_retry",
            )

            # Submit the failing task that should be retried
            handler.handle_amqp_request(channel_retry, method_retry, props_retry, body_retry)

            # Wait for retries to complete
            time.sleep(15)

            # Verify task was dropped after retries (custom predicate allowed retries)
            assert len(handler.running_tasks) == 0, "Retry task should be dropped after retries"

            # Verify retry attempts were made (should see error about failed retries)
            mock_error = mock_logger.error
            retry_error_calls = [call for call in mock_error.call_args_list if "failed after" in str(call)]
            assert len(retry_error_calls) == 1, "Should log error message about failed retries for retry task"
            assert "2 retries" in str(retry_error_calls[0])

            # Test Case 7b: Task that should NOT be retried according to custom predicate
            channel_no_retry, method_no_retry, props_no_retry, body_no_retry = create_amqp_request_mock(
                routing_key="noop_fail",
                body="no_retry_task",
                reply_to="test_reply_no_retry",
                correlation_id="correlation_no_retry",
            )

            # Submit the failing task that should NOT be retried
            handler.handle_amqp_request(channel_no_retry, method_no_retry, props_no_retry, body_no_retry)

            # Wait for processing to complete (should fail quickly without retries)
            time.sleep(0.1)

            # Verify task was dropped quickly (custom predicate prevented retries)
            assert len(handler.running_tasks) == 0, "No-retry task should be dropped quickly"

            # Verify that the custom predicate prevented retries by checking for the specific log message
            mock_error = mock_logger.error
            no_retry_error_calls = [
                call for call in mock_error.call_args_list if "failed after 0 retries." in str(call)
            ]
            assert len(no_retry_error_calls) == 1, "Should log info message about retry predicate failure"

            # Verify no replies were sent for either failed task
            mock_amqp_client.reply.assert_not_called()
        finally:
            # Clean up
            handler.shutdown()

    def test_two_long_tasks_processing(self, handler, mock_amqp_client):
        """Test Case 6: Check that two long tasks are processed correctly without deadlock"""

        # Create a task that will just wait
        channel, method, props, body = create_amqp_request_mock(routing_key="noop_sleep", body=2)

        # Submit the long-running task
        handler.handle_amqp_request(channel, method, props, body)

        # Wait for task to be picked up and record original start time
        time.sleep(0.02)

        # Create another task
        channel, method, props, body = create_amqp_request_mock(routing_key="noop_sleep", body=3)

        # Submit the long-running task once again
        handler.handle_amqp_request(channel, method, props, body)

        # Wait for task to be picked up and record original start time
        time.sleep(0.02)

        # Create the last task
        channel, method, props, body = create_amqp_request_mock(routing_key="noop_sleep", body=3)

        # Submit it
        handler.handle_amqp_request(channel, method, props, body)

        # Wait for task to be picked up and record original start time
        time.sleep(0.02)

        # Verify task is in running_tasks and record start time
        assert len(handler.running_tasks) == 2, "A task should appear in running_tasks"
        assert handler.queue.qsize() == 1, "The last task should be queued"

        # Wait for all tasks to complete
        time.sleep(15)

        # Verify task was dropped after retries
        assert len(handler.running_tasks) == 0, "Tasks should be processed and removed from running_tasks"
        assert handler.queue.qsize() == 0, "Queue should be empty after processing all tasks"
