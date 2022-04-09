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
                    "number_of_shards":2,
                    "refresh_interval" : "-1",
                    "knn" : "true",
                },
            },
            "mappings":{
                "properties":{
                    "recipeId":{
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
                    "sentence_embedding_title": {
                        "type":"knn_vector",
                        "dimension": 768,
                        "method":{
                            "name":"hnsw",
                            "space_type":"innerproduct",
                            "engine":"faiss",
                            "parameters":{
                                "ef_construction":256,
                                "m":48
                            }
                        }
                    },
                    "sentence_embedding_description": {
                        "type":"knn_vector",
                        "dimension": 768,
                        "method":{
                            "name":"hnsw",
                            "space_type":"innerproduct",
                            "engine":"faiss",
                            "parameters":{
                                "ef_construction":256,
                                "m":48
                            }
                        }
                    },
                    "ingredients":{
                        "type":"keyword"
                    },
                    "keywords":{
                        "type":"keyword"
                    },
                    "time":{
                        "type":"integer"
                    }
                }
            }
        }

        if not self.client.indices.exists(self.index_name):
            response = self.client.indices.create(self.index_name, body=index_body)
            print('\nCreating index:')
            print(response)
        
        self.updateIndexSettings()
        return



    def updateIndexSettings(self):
        index_settings = {
            "settings":{
                "index":{
                    "refresh_interval" : "1s"
                }
            }
        }
        self.client.indices.put_settings(index = self.index_name, body = index_settings)