from jsonsocket import Client, Server

host = 'localhost'
port = 8000

# Client code:
client = Client()
#client.connect(host, port).send({'some_list': [123, 456]})
client.connect(host, port).send([123, 456])
response = client.recv()
print('response: ', response)
# response now is {'data': {'some_list': [123, 456]}}
client.close()
