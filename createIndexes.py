import pprint as pp


class CreateIndexes():
    def __init__(self, client, index_name):
        self.client = client
        self.index_name = index_name

    def createIndexStructure(self):
        index_body = {
            "settings":{
                "index":{
                    "number_of_replicas":0,
                    "number_of_shards":4,
                    "refresh_interval" : "-1",
                    "knn" : "true",
                },
            },
            "mappings":{
                "properties":{
                    "recipesId":{
                        "type":"keyword"
                    },
                    "title":{
                        "type":"text",
                        "similarity":"BM25"
                    },
                    "description":{
                        "type":"text",
                        "similarity":"BM25"
                    },
                    "sentence_embedding_NQ": {
                    "type": "knn_vector",
                    "model_id": "model_kwiz"
                    },
                    "sentence_embedding_NQ_FT": {
                    "type": "knn_vector",
                    "model_id": "model_kwiz"
                    }
                }
            }
        }

        if not self.client.indices.exists(self.index_name):
            response = self.client.indices.create(self.index_name, body=index_body)
            print('\nCreating index:')
            print(response)

        return



    def updateIndexSettings(self):
        print('\n----------------------------------------------------------------------------------- INDEX SETTINGS')
        index_settings = {
            "settings":{
                "index":{
                    "refresh_interval" : "1s"
                }
            }
        }
        self.client.indices.put_settings(index = self.index_name, body = index_settings)
        settings = self.client.indices.get_settings(index = self.index_name)
        pp.pprint(settings)

        print('\n----------------------------------------------------------------------------------- INDEX MAPPINGS')
        mappings = self.client.indices.get_mapping(index = self.index_name)
        pp.pprint(mappings)

        print('\n----------------------------------------------------------------------------------- INDEX #DOCs')
        print(self.client.count(index = self.index_name))