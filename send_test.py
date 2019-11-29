import pika
import sys
import uuid
import json

class RpcClient(object):

    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.connection.\
        URLParameters("amqp://rabbitmq:rabbitmq@localhost:5672/analyzer?heartbeat=600"))

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, message, method):
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
        "indexingRunning": True
    },
    "testItems": [
        {"testItemId": 2,
        "uniqueId": "df",
        "isAutoAnalyzed": False,
        "issueType": "pb001",
        "originalIssueType": "PB001",
        "logs": [
            {"logId": 3,
            "logLevel": 40000,
            "message": "error occured"}
        ]}
    ]
}]

search_data = {
        "launchId": 4,
        "launchName": "dsfdsf",
        "itemId": 3,
        "projectId": 34,
        "filteredLaunchIds": [1],
        "logMessages": ["error occured"],
        "logLines": -1
    }

clean_index_data = {
    "ids": [3],
    "project": 34
}

method = sys.argv[1] if len(sys.argv) > 0 else "index"
print(" [x] calling method %s"%method)
if method.strip() in ["delete"]:
    rpc.call_without_wait("34", method)
    print("Method '%s' was called"%method)
elif method.strip() in ["clean"]:
    rpc.call_without_wait(json.dumps(clean_index_data), method)
    print("Method '%s' was called"%method)
elif method.strip() in ["search"]: 
    response = rpc.call(json.dumps(search_data), method)
    print(" [.] Got %r" % response)
else:
    response = rpc.call(json.dumps(index_data), method)
    print(" [.] Got %r" % response)