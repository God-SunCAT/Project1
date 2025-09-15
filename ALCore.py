import json
import textwrap
from module.VectorDB import SimpleVectorDB
from module.LlamaRequest import llm_ask, llm_embedding

class AAL:
    def __init__(self):
        # self.db = SimpleVectorDB(1024, 10000, "./db/VectorDB")
        pass
    
    def ask(self, context):
        return 'Hello'
    
    def selfModeling(self, context):
        # 注意：context表示上下文，content表示回复内容
        '''
        自我建模：
            1.根据一组对话对其产生问题
            2.分别回答这些问题
            3.写入向量数据库
        '''
        # 根据一组对话对其产生问题
        promote = textwrap.dedent(f'''
            你是一个自我建模机器，请对如下文字做分析，对文本提出多条关于“我”(文字撰写者)的问题。
            自行按内容划分不同提问维度，每个维度只取最重要的问题。且问题数不得超过5条，并以严格的json输出（只输出json，不得输出其他内容）。
            示例：
            [
            {{"content": "你最想/最害怕在自我认知上发现或承认的，究竟是什么？"}},
            {{"content": "你的这个“最高准则”具体包含哪些核心信念或禁忌，它如何指导你在感情与生活中的决断？"}},
            {{"content": "在爱与占有之间，你如何衡量保护对方和尊重对方自由的界限？"}},
            ]
            本次任务文本：
            {context}
        ''')
        answer = llm_ask(promote)
        # 回答问题
        cognition = []
        data = json.loads(answer)
        if(len(data) == 0):
            print('Error: Self-Modeling Question Failed.')
            return
        
        for item in data:
            if(not 'content' in item):
                print('Error: Self-Modeling Answer Failed.')
                return
            
            promote = textwrap.dedent(f'''
                请使用下面的文本回复该问题：{item['content']}
                要求以描述性的语句回答，简短精要，尽量不适用复杂华丽的修饰，要求以白话形式描述，而非文学语言。
                每条context必须具备完备性，其自身就具备所有必须内容，不需要其他上下文补充解释。
                并且，每条context必须包含“我”，如"我认为"、"我最想发现的是"等。
                特别注意，不可使用"你""他们"这样的抽象代词，必须使用具体指代如姓名、组织名、具体关系名（请自行根据上下文推断指代）。
                回复应该包含问题的关键内容,并以严格的json输出（只输出json，不得输出其他内容）。
                示例：
                问题：你最想/最害怕在自我认知上发现或承认的，究竟是什么？
                回复：
                [
                {{"content": "在自我认知上我最想发现的是：自己的内心一直在翻腾，不会满足、会保持追求。"}},
                {{"content": "在自我认知上我最害怕承认的是：我其实很自私，或者有一天变得心安理得、沉默不前。"}},
                ]
                本次任务文本：
                {context}
            ''')
            answer = llm_ask(promote)

            data2 = json.loads(answer)
            if(len(data2) == 0):
                print('Error: Self-Modeling Answer Failed.')
                return
        
            for item in data2:
                if(not 'content' in item):
                    return
                cognition.append(item['content'])
            
        # 写入向量数据库
        print(cognition)
            
        
core = AAL()
core.selfModeling(
'''
银河：

你好！昨天我写了一封卑鄙的信，你一定伤心了。我太不对了。今天我痛悔不已。

我怎么能背弃你呢。你是那么希望我成长起来，摆脱无所事事的软弱。我现在一步也离不开你，不然我又要不知做什么好了。

我很难过的是，你身边那么多人都对我反目而视。我并不太坏呀。我要向你靠拢，可是一个人的惰性不是那么好克服的。有时我要旧态复萌，然后就后悔。你想，我从前根本不以为我可以合上社会潮流的节拍，现在不是试着去做了吗？我这样的人试一试就要先碰上几鼻子灰，不是情所当然的吗？我真的决心放弃以前的一切，只要你说怎么办。你又不说。

我真的不知怎么才能和你亲近起来，你好像是一个可望而不可及的目标，我捉摸不透，追也追不上，就坐下哭了起来。

算了，不多说。我只求你告诉我，我到底能不能得到你。我还不算太笨，还能干好多事情。你告诉我怎么办吧
'''
)