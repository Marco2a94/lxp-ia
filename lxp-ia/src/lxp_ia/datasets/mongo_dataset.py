from kedro.io import AbstractDataset
from pymongo import MongoClient
import pandas as pd


class MongoDataset(AbstractDataset):
    def __init__(self, uri, db, collection):
        self.uri = uri
        self.db = db
        self.collection = collection

    def _load(self):
        client = MongoClient(self.uri)
        data = list(
            client[self.db][self.collection].find({}, {"_id": 0})
        )
        return pd.DataFrame(data)

    def _save(self, data):
        raise NotImplementedError("Read-only dataset")

    def _describe(self):
        return {
            "uri": self.uri,
            "db": self.db,
            "collection": self.collection,
        }
