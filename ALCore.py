from module.VectorDB import SimpleVectorDB
from module.LlamaRequest import llm_ask, llm_embedding
class AAL:
    def __init__(self):
        self.db = SimpleVectorDB(1024, 10000, "./db/VectorDB")
        self.db.add(llm_embedding("a"), "a")
    
    def ask(self, context):
        return 'Hello'