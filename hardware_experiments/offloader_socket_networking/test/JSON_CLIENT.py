# default uses python 2
import sys,os
base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)

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
