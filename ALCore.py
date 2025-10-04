import json
import time
import os
import pickle
import logging
from NLPN import NLPN
from module.Prompts import *
from module.VectorDB import SimpleVectorDB, queryByWeight
from module.LlamaRequest import llm_ask, llm_embedding

# 注意，标准的json最后一项没有 , 并且，只使用"
class AAL:
    def __init__(self):
        self.net = NLPN()

        # 配置向量数据库
        self.SelfDB = SimpleVectorDB(1024, 10000, "./db/SelfModeling_VectorDB")
        self.MemDB = SimpleVectorDB(1024, 10000, "./db/DetailMemory_VectorDB")
        self.ComMemDB = SimpleVectorDB(1024, 10000, "./db/CompressionMemory_VectorDB")

        # 读取配置数据
        if os.path.exists('./db/data.pkl'):
            with open('./db/data.pkl', "rb") as f:
                self.conf = pickle.load(f)
        else:
            # 其ID表示已读的最后一条
            # History [('role', 'content'), ...]
            self.conf = {'self': 0, 'mem': 0, 'history': []}
    
    def ask(self, message, userName='User-001', lifeName='伊芙'):
        # 注:FNode = -1时会被忽略

        # Query-Rewriting
        t = time.time()

        answer = llm_ask(pmt_ASK_QuestionSplit.format(context=message))
        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Ask-Question Split Failed.')
            return 'Error'
        
        logging.info(
            '----\n'
            'Time:Query-Rewriting:'
            f'{(time.time() - t):.2f}秒\n'
            '----\n'
        )
        
        # VectorDB-Query
        t = time.time()

        result_Self = []
        result_Mem = []
        result_Cog = []
        result_list = []

        FNodes = []
        for item in data:
            if('self' in item):
                # Self-Modeling数据库
                # x = self.SelfDB.query(llm_embedding(item['self']))
                x = queryByWeight(self.SelfDB, llm_embedding(item['self']), 5)
                FNodes += [i[0]['fnode'] for i in x if i[0] is not None]
                # [(content, fnode, id, latest, DBType), ...]
                # 注 latest = nextID - 1 所以这里已经预留出history的timeStep了
                result_Self += [(i[0]['content'], i[0]['fnode'], i[1], self.SelfDB.next_id, 'self') for i in x if i[0] is not None]
            elif('mem' in item):
                # x = self.MemDB.query(llm_embedding(item['mem']))
                x = queryByWeight(self.MemDB, llm_embedding(item['mem']), 5)
                FNodes += [i[0]['fnode'] for i in x if i[0] is not None]
                result_Mem += [(i[0]['content'], i[0]['fnode'], i[1], self.MemDB.next_id, 'detailMem') for i in x if i[0] is not None]
        result_Self = set(result_Self)
        result_Mem = set(result_Mem)
        # set 直接相加是非法操作，应该用并集运算
        result = result_Self | result_Mem

        logging.info(
            '----\n'
            'Time:VectorDB-Query:'
            f'{(time.time() - t):.2f}秒\n'
            '----\n'
        )

        # 构建 概括记忆与细节记忆 依赖
        # [{'node': 1, 'mem': '', 'detail': []}]
        # [{'mem': ('content', time), 'detail': [('content', time, type)]}]
        # (timeStep/latest) - (latest-timeStep) - (rate)
        t = time.time()
        
        # 时间步格式化函数
        getTime = lambda dataTime, latestTime: f'{dataTime}/{latestTime} - {latestTime - dataTime} - {((dataTime/latestTime) * 100):.2f}%'
        # 数据序列化
        FNodes = set(FNodes)
        for i in FNodes:
            if(i == -1):
                continue
            detail = []
            for text, node, dataTime, latestTime, DBType in result:
                if node == i:
                    detail.append((text, getTime(dataTime, latestTime), DBType))
            result_list.append(
                {'mem': (self.ComMemDB.query_by_id(i)['content'], getTime(i, self.ComMemDB.next_id)), 'detail': detail}
            )

        logging.info(
            '----\n'
            'Time:MakeList-Mem&detail:'
            f'{(time.time() - t):.2f}秒\n'
            '----\n'
        )
        # Mem-Refine
        t = time.time()

        # 修改，应使得历史记录仅使用用户提问。以免造成自身记忆污染(要污染也要在NLPN中污染！)。
        history = \
            f'TimeStep - {self.ComMemDB.next_id - 2}\n' + \
            f'{self.ComMemDB.query_by_id(self.ComMemDB.next_id - 2)}\n' + \
            f'TimeStep - {self.ComMemDB.next_id - 1}\n' + \
            f'{self.ComMemDB.query_by_id(self.ComMemDB.next_id - 1)}\n' + \
            f'TimeStep - {self.ComMemDB.next_id}\n' + \
            f'{str([i for i in self.conf["history"] if i[0] == userName])}\n'

        answer = llm_ask(
            pmt_ASK_MemRefine.format(question=message, context=str(result_list), history=history),
            mode='high'
        )

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Mem-Compression Failed.')
            return 'Error'
        
        logging.info(
            '----\n'
            'Time:Mem-Refine:'
            f'{(time.time() - t):.2f}秒\n'
            '----\n'
        )
        # Humanization
        # 对于记忆系统结果的后处理我是真没辙了，完全没想过怎么使用记忆。
        # t = time.time()

        # answer = llm_ask(pmt_ASK_Humanization.format(question=message, context=data[0]['content']), mode='high')

        # data = json.loads(answer.strip())
        # if(len(data) == 0):
        #     print('Error: Humanization Failed.')
        #     return 'Error'
        
        # logging.info(
        #     '----\n'
        #     'Time:Humanization:'
        #     f'{(time.time() - t):.2f}秒\n'
        #     '----\n'
        # )

        # 保存历史记录
        self.conf['history'].append((userName, message))
        self.conf['history'].append((lifeName, data[0]['content']))

        with open("./db/data.pkl", "wb") as f:  # wb = write binary
            pickle.dump(self.conf, f)

        return data[0]['content']


    def selfModeling(self, message, lifeName='伊芙'):
        # 注意：context表示上下文，content表示回复内容
        '''
        自我建模：
            1.根据一组对话对其产生问题
            2.分别回答这些问题
            3.写入向量数据库
        '''
        # Compression 先完成对概括类记忆的构建
        answer = llm_ask(pmt_MEM_Compression.format(person=lifeName, context=message))
        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Make-CompressionMemory Failed.')
            return -1, -1, -1
        
        FNode = self.ComMemDB.add(llm_embedding(data[0]['content']), data[0])

        # Question
        answer = llm_ask(pmt_SM_Question.format(person=lifeName, context=message))
        
        # Answer
        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Self-Modeling Question Failed.')
            return -1, -1, -1
        cognition = ''
        for item in data:
            if(not 'content' in item):
                print('Error: Self-Modeling Answer Failed.')
                return -1, -1, -1
            answer = llm_ask(pmt_SM_Answer.format(question=item['content'], context=message, person=lifeName))
            
            data2 = json.loads(answer.strip())
            if(len(data2) == 0):
                print('Error: Self-Modeling Answer Failed.')
                return -1, -1, -1
        
            for item in data2:
                if(not 'content' in item or not 'weight' in item):
                    return -1, -1, -1
                cognition += str(item) + "\n"
                item['fnode'] = FNode
                lastIDSelf = self.SelfDB.add(llm_embedding(item['content']), item)
                
        # 事件记忆建模
        answer = llm_ask(pmt_MEM_Modeling.format(person=lifeName, cognition=cognition, context=message))

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Memory-Modeling Answer Failed.')
            return -1, -1, -1

        for item in data:
            if(not 'content' in item or not 'weight' in item):
                return -1, -1, -1
            item['fnode'] = FNode
            lastIDMem = self.MemDB.add(llm_embedding(item['content']), item)

        # NLPN
        n = 30
        if(lastIDSelf - self.conf['self'] >= n):
            # 读取最新的Self数据和Mem数据，并完成格式化
            # AuxiliaryData -> (text-list, embedding-list)
            # SourceData -> (text-list, embedding-list)
            logging.info(
                '----\n'
                '<NLPN-SelfModeling>\n'
                f'[{self.conf.get("self", 0) + 1}, {lastIDSelf}]\n'
                '----\n'
            )
            # 构造AuxiliaryData
            texts = []
            embeddings = []
            for i in range(self.conf['mem'] + 1, lastIDMem + 1, 1):
                x = self.MemDB.query_by_id(i, True)
                texts.append(x[1]['content'])
                embeddings.append(x[0])
            aux = (texts[:], embeddings[:])

            # 构造SourceData
            texts = []
            embeddings = []
            for i in range(self.conf['self'] + 1, lastIDSelf + 1, 1):
                x = self.SelfDB.query_by_id(i, True)
                texts.append(x[1]['content'])
                embeddings.append(x[0])
            source = (texts[:], embeddings[:])
            lastID = self.net.Modeling(aux, source, self.SelfDB)

            self.conf['self'] = lastID
            self.conf['mem'] = lastIDMem
            with open("./db/data.pkl", "wb") as f:  # wb = write binary
                pickle.dump(self.conf, f)
        return self.SelfDB.next_id - 1, self.MemDB.next_id - 1, self.ComMemDB.next_id - 1


