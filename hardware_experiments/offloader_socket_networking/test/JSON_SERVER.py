# default uses python 2

import sys,os
base_video_dir = os.environ['CLOUD_ROOT_DIR']
utils_dir = base_video_dir + '/utils/'
sys.path.append(utils_dir)


from jsonsocket import Client, Server

host = 'localhost'
port = 8000

# Server code:
server = Server(host, port)
server.accept()
data = server.recv()
# data now is: {'some_list': [123, 456]}
server.send({'data': data})
print('SENT DATA')
server.close()
print('CLOSED')
