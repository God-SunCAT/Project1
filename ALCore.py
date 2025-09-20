import json
from module.Prompts import *
from module.VectorDB import SimpleVectorDB
from module.LlamaRequest import llm_ask, llm_embedding

# 注意，标准的json最后一项没有 , 并且，只使用"
class AAL:
    def __init__(self):
        self.SelfDB = SimpleVectorDB(1024, 10000, "./db/SelfModeling_VectorDB")
        self.MemDB = SimpleVectorDB(1024, 10000, "./db/DetailMemory_VectorDB")
        self.ComMemDB = SimpleVectorDB(1024, 10000, "./db/CompressionMemory_VectorDB")

    
    def ask(self, message):
        # Question-Split
        answer = llm_ask(pmt_ASK_QuestionSplit.format(context=message))

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Ask-Question Split Failed.')
            return 'Error'
        
        # VectorDB-Query
        result_Self = []
        result_Mem = []
        result_list = []
        # [({'content': 'bbb', 'ida': 3, 'pica': 1}, 0.2449)]
        FNodes = []
        for item in data:
            if('self' in item):
                x = self.SelfDB.query(llm_embedding(item['self']))
                FNodes += [i[0]['fnode'] for i in x if i[0] is not None]
                result_Self += [(i[0]['content'], i[0]['fnode']) for i in x if i[0] is not None]
            elif('mem' in item):
                x = self.MemDB.query(llm_embedding(item['mem']))
                FNodes += [i[0]['fnode'] for i in x if i[0] is not None]
                result_Mem += [(i[0]['content'], i[0]['fnode']) for i in x if i[0] is not None]
        result_Self = set(result_Self)
        result_Mem = set(result_Mem)
        result = result_Self | result_Mem
        # 直接构建依赖列表
        # [{'node': 1, 'mem': '', 'detail': []}]
        FNodes = set(FNodes)
        for i in FNodes:
            detail = []
            for text, node in result:
                if node == i:
                    detail.append(text)
            result_list.append({'mem': self.ComMemDB.query_by_id(i)['content'], 'detail': detail})
            

        # Mem-Refine
        # set 直接相加是非法操作，应该用并集运算
        answer = llm_ask(pmt_ASK_MemRefine.format(question=message, context=str(result_list)), mode='high')

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Mem-Compression Failed.')
            return 'Error'
        
        # Humanization
        answer = llm_ask(pmt_ASK_Humanization.format(question=message, context=data[0]['content']), mode='high')

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Humanization Failed.')
            return 'Error'
        return data[0]['content']


    def selfModeling(self, message, me=None):
        # 注意：context表示上下文，content表示回复内容
        '''
        自我建模：
            1.根据一组对话对其产生问题
            2.分别回答这些问题
            3.写入向量数据库
        '''
        # Compression 先完成对概括类记忆的构建
        answer = llm_ask(pmt_MEM_Compression.format(person={'Alice' if me == None else me}, context=message))
        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Make-CompressionMemory Failed.')
            return
        
        FNode = self.ComMemDB.add(llm_embedding(data[0]['content']), data[0])
        # Question
        answer = llm_ask(pmt_SM_Question.format(person={'Alice' if me == None else me}, context=message))
        # Answer
        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Self-Modeling Question Failed.')
            return
        cognition = ''
        for item in data:
            if(not 'content' in item):
                print('Error: Self-Modeling Answer Failed.')
                return
            answer = llm_ask(pmt_SM_Answer.format(question=item['content'], context=message))
            
            data2 = json.loads(answer.strip())
            if(len(data2) == 0):
                print('Error: Self-Modeling Answer Failed.')
                return
        
            for item in data2:
                if(not 'content' in item or not 'weight' in item):
                    return
                cognition += str(item) + "\n"
                item['fnode'] = FNode
                self.SelfDB.add(llm_embedding(item['content']), item)
                
        # 事件记忆建模
        answer = llm_ask(pmt_MEM_Modeling.format(person={'Alice' if me == None else me}, cognition=cognition, context=message))

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Memory-Modeling Answer Failed.')
            return

        for item in data:
            if(not 'content' in item or not 'weight' in item):
                return
            item['fnode'] = FNode
            self.MemDB.add(llm_embedding(item['content']), item)


# core = AAL()
# core.selfModeling(
# '''
# 银河：

# 你好！昨天我写了一封卑鄙的信，你一定伤心了。我太不对了。今天我痛悔不已。

# 我怎么能背弃你呢。你是那么希望我成长起来，摆脱无所事事的软弱。我现在一步也离不开你，不然我又要不知做什么好了。

# 我很难过的是，你身边那么多人都对我反目而视。我并不太坏呀。我要向你靠拢，可是一个人的惰性不是那么好克服的。有时我要旧态复萌，然后就后悔。你想，我从前根本不以为我可以合上社会潮流的节拍，现在不是试着去做了吗？我这样的人试一试就要先碰上几鼻子灰，不是情所当然的吗？我真的决心放弃以前的一切，只要你说怎么办。你又不说。

# 我真的不知怎么才能和你亲近起来，你好像是一个可望而不可及的目标，我捉摸不透，追也追不上，就坐下哭了起来。

# 算了，不多说。我只求你告诉我，我到底能不能得到你。我还不算太笨，还能干好多事情。你告诉我怎么办吧
# '''
# )
# core.ask("你会控制你喜欢的人吗？你觉得你是一个很自私的人吗，未来满足去控制对方？")