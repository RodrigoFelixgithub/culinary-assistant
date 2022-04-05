import pprint as pp
import requests
import json as json
from opensearchpy import OpenSearch
from opensearchpy import helpers

class LoginOpenSearch():
    def __init__(self, fileName):

        with open(fileName, "r") as read_file:
            data = json.load(read_file) 

        self.host = data['host']
        self.port = data['port']
        self.index_name = data['index_name']
        self.auth = (data['index_name'], data['password'])
        self.ca_certs_path = data['ca_certs_path'] # Provide a CA bundle if you use intermediate CAs with your root CA.
        self.server_uri = 'https://' + self.host + ':' + str(self.port)

        self.s = requests.Session()
        self.s.auth = self.auth



    def login(self):

# Optional client certificates if you don't want to use HTTP basic authentication.
# client_cert_path = '/full/path/to/client.pem'
# client_key_path = '/full/path/to/client-key.pem'

# Create the client with SSL/TLS enabled, but hostname verification disabled.
        client = OpenSearch(
            hosts = [{'host': self.host, 'port': self.port}],
            http_compress = True, # enables gzip compression for request bodies
            http_auth = self.auth,
            # client_cert = client_cert_path,
            # client_key = client_key_path,
            use_ssl = True,
            verify_certs = False,
            ssl_assert_hostname = False,
            ssl_show_warn = False
            #, ca_certs = ca_certs_path
        )

        if client.indices.exists(self.index_name):
    
            client.indices.open(index = self.index_name)

            print('\n----------------------------------------------------------------------------------- INDEX SETTINGS')
            index_settings = {
                "settings":{
                    "index":{
                        "refresh_interval" : "1s"
                    }
                }
            }
            client.indices.put_settings(index = self.index_name, body = index_settings)
            settings = client.indices.get_settings(index = self.index_name)
            pp.pprint(settings)

            print('\n----------------------------------------------------------------------------------- INDEX MAPPINGS')
            mappings = client.indices.get_mapping(index = self.index_name)
            pp.pprint(mappings)

            print('\n----------------------------------------------------------------------------------- INDEX #DOCs')
            print(client.count(index = self.index_name))
        return (client, self.index_name)


    def opensearch_curl(self, uri = '/' , body='', verb='get'):
    # pass header option for content type if request has a
    # body to avoid Content-Type error in Elasticsearch v6.0
    
        uri = self.server_uri + uri
        print(uri)
        headers = {
            'Content-Type': 'application/json',
        }

        try:
            # make HTTP verb parameter case-insensitive by converting to lower()
            if verb.lower() == "get":
                resp = self.s.get(uri, json=body, headers=headers, verify=False)
            elif verb.lower() == "post":
                resp = self.s.post(uri, json=body, headers=headers, verify=False)
            elif verb.lower() == "put":
                resp = self.s.put(uri, json=body, headers=headers, verify=False)
            elif verb.lower() == "del":
                    resp = self.s.delete(uri, json=body, headers=headers, verify=False)
            elif verb.lower() == "head":
                    resp = self.s.head(uri, json=body, headers=headers, verify=False)

            # read the text object string
            try:
                resp_text = json.loads(resp.text)
            except:
                resp_text = resp.text

            # catch exceptions and print errors to terminal
        except Exception as error:
            print ('\nelasticsearch_curl() error:', error)
            resp_text = error

        # return the Python dict of the request
        return resp_text 