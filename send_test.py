"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import sys
import uuid
import json
import pika

# This file can be used for checking app functionality locally


class RpcClient():
    """RpcClient helps to use RPC type of communication with rabbitmq"""
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.connection.
            URLParameters("amqp://rabbitmq:rabbitmq@localhost:5672/analyzer?heartbeat=600"))
        self.response = None
        self.corr_id = None

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=lambda channel, method, props, body: self._on_response(props, body),
            auto_ack=True)

    def _on_response(self, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, message, method):
        """RpcClient sends a message to a queue and waits for the answer"""
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='analyzer',
            routing_key=method,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=message)

        while self.response is None:
            self.connection.process_data_events()
        return json.loads(self.response)

    def call_without_wait(self, message, method):
        """RpcClient sends a message to a queue and doesn't wait for the answer"""
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='analyzer',
            routing_key=method,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=message)


rpc = RpcClient()

index_data = [{
    "launchId": 1,
    "project": 34,
    "launchName": "dsfdsf",
    "analyzerConfig": {
        "minDocFreq": 1.0,
        "minTermFreq": 1.0,
        "minShouldMatch": 80,
        "numberOfLogLines": -1,
        "isAutoAnalyzerEnabled": True,
        "analyzerMode": "ALL",
        "indexingRunning": True,
    },
    "testItems": [{"testItemId": 2,
                   "uniqueId": "df",
                   "isAutoAnalyzed": False,
                   "issueType": "pb001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 3,
                        "logLevel": 40000,
                        "message": "error occured"},
                       {"logId": 4,
                        "logLevel": 40000,
                        "message": "error occured \r\n error found \r\n error mined"}, ]
                   },
                  {"testItemId": 5,
                   "uniqueId": "df1",
                   "isAutoAnalyzed": False,
                   "issueType": "ti001",
                   "originalIssueType": "TI001",
                   "logs": [
                       {"logId": 5,
                        "logLevel": 40000,
                        "message": "error occured \r\n error found \r\n error mined"}, ]
                   }, ],
}, {
    "launchId": 1,
    "project": 34,
    "launchName": "dsfdsf",
    "analyzerConfig": {
        "minDocFreq": 1.0,
        "minTermFreq": 1.0,
        "minShouldMatch": 80,
        "numberOfLogLines": -1,
        "isAutoAnalyzerEnabled": True,
        "analyzerMode": "ALL",
        "indexingRunning": True,
    },
    "testItems": [{"testItemId": 5,
                   "uniqueId": "df5",
                   "isAutoAnalyzed": False,
                   "issueType": "ab001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 6,
                        "logLevel": 40000,
                        "message": "error occured"},
                       {"logId": 8,
                        "logLevel": 40000,
                        "message": "error occured \r\n error found \r\n error mined"}, ]
                   },
                  {"testItemId": 78,
                   "uniqueId": "df5",
                   "isAutoAnalyzed": False,
                   "issueType": "pb001",
                   "originalIssueType": "PB001",
                   "logs": [
                       {"logId": 45,
                        "logLevel": 40000,
                        "message": "error occured"},
                       {"logId": 81,
                        "logLevel": 40000,
                        "message": "error occured \r\n error found \r\n error mined"}, ]
                   },
                  {"testItemId": 10,
                   "uniqueId": "df12",
                   "isAutoAnalyzed": False,
                   "issueType": "ab001",
                   "originalIssueType": "ab001",
                   "logs": [
                       {"logId": 38,
                        "logLevel": 40000,
                        "message": "error occured \r\n error found \r\n error mined"}, ]
                   }, ],
}]

search_data = {
    "launchId": 4,
    "launchName": "dsfdsf",
    "itemId": 3,
    "projectId": 34,
    "filteredLaunchIds": [1],
    "logMessages": ["error occured found mined", ],
    "logLines": -1, }

clean_index_data = {
    "ids": [3],
    "project": 34,
}

test_item_info = {
    "testItemId": 4,
    "uniqueId": "unique",
    "testCaseHash": 111,
    "launchId": 3,
    "launchName": "Launch name",
    "project": 34,
    "analyzerConfig": {
        "minDocFreq": 1.0,
        "minTermFreq": 1.0,
        "minShouldMatch": 80,
        "numberOfLogLines": -1,
        "isAutoAnalyzerEnabled": True,
        "analyzerMode": "ALL",
        "indexingRunning": True,
    },
    "logs": [{"logId": 3,
              "logLevel": 40000,
              "message": "error occured"},
             {"logId": 4,
              "logLevel": 40000,
              "message": "error occured \r\n error found \r\n error mined"}]
}

used_method = sys.argv[1] if len(sys.argv) > 1 else "index"
print(" [x] calling method %s" % used_method)
if used_method.strip() in ["delete"]:
    response = rpc.call("34", used_method)
elif used_method.strip() in ["clean"]:
    response = rpc.call(json.dumps(clean_index_data), used_method)
elif used_method.strip() in ["search"]:
    response = rpc.call(json.dumps(search_data), used_method)
elif used_method.strip() in ["suggest"]:
    response = rpc.call(json.dumps(test_item_info), used_method)
elif used_method.strip() in ["cluster"]:
    response = rpc.call(json.dumps({"launch": index_data[0], "for_update": "false"}), used_method)
else:
    response = rpc.call(json.dumps(index_data), used_method)
print(" [.] Got %r" % response)
