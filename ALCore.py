import json
import numpy as np
from module.VectorDB import SimpleVectorDB
from module.LlamaRequest import llm_ask, llm_embedding

# 注意，标准的json最后一项没有 , 并且，只使用"
class AAL:
    def __init__(self):
        self.SelfDB = SimpleVectorDB(1024, 10000, "./db/SelfModeling_VectorDB")
        self.MemDB = SimpleVectorDB(1024, 10000, "./db/Memory_VectorDB")
    
    def ask(self, context):
        promote = f'''
你是一个问题拆分机器，将输入文本视为一个大的问题。自行思考，如果作为人类回复这个文本，你会去回忆哪些内容？
为其生成多条问题，以便通过向量数据库寻找记忆。请自行调整问题数量分配，总数量不得超过5个。
请根据输入内容的具体程度来决定问题的具体程度，若输入内容为比较宽泛的内容，输出的问题也不要过于具体(向量数据库没有庞大到能够响应全部类型的提问)。
问题分为两类，一类用于自我建模数据库，一类用于记忆建模数据库。
自我建模数据库偏向抽象化、自我认知层面的总结，其强调“我”对自身心理状态的分析，其多使用如"我认为"、"我最想发现的是"等词语。
其记录来自对自我剖析问题的回答(示例)：
1.我渴望得到别人的认可，但不确定自己是否值得被接纳。
2.我依赖对方的指引来确认自己是否值得被接纳，这种依赖让我既渴望靠近又感到挫败。
记忆建模数据库则主要偏向对实际事件与行为的记录，其以客观的语言(使用第一人称代词“我”)和角度表述客观存在的事件或“我”所认为的事件。
其记录来自于对接受输入的总结，内容示例如下：
1.昨天我写了一封卑鄙的信，并认为银河因此伤心，我为此感到后悔并承认错误。
2.我强调自己“还不算太笨、还能干很多事”，这是我在表述依赖与求助时保留的自我能力认知。
输出示例(要求以严格的JSON示例输出，不得输出其他任何内容。self表示自我建模，mem表示记忆建模)：
[
{{"self": "我是否渴望通过控制他人来获得自我价值感？"}},
{{"mem": "我是否曾试图通过控制他人来满足自己的情感需求？"}}
]
本次任务文本：
{context}
'''
        answer = llm_ask(promote)

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Ask-Question Split Failed.')
            return 'Error'
        result_Self = []
        result_Mem = []
        # [
        #     ({'content': 'bbb', 'ida': 3, 'pica': 1}, 0.2449),
        #     ({'content': 'aaa', 'ida': 2}, 0.2528),
        #     ({'content': 'ccc', 'ida': 2}, 0.2649)
        # ]
        for item in data:
            if('self' in item):
                x = self.SelfDB.query(llm_embedding(item['self']))
                result_Self += [i[0]['content'] for i in x if i[0] is not None]
            elif('mem' in item):
                x = self.MemDB.query(llm_embedding(item['mem']))
                result_Mem += [i[0]['content'] for i in x if i[0] is not None]
        result_Self = set(result_Self)
        result_Mem = set(result_Mem)
        promote = f'''
你是一个记忆压缩机器，请根据问题自行考虑各种信息对问题的重要程度。
你需要将给出的文本精炼为只对回答特定问题重要一段文字。
要求该段文字以客观的语言(使用第一人称代词“我”)和角度表述内容，并且要求自行按需保留原文本所流露出的情感。
要求以描述性的语句回答，简短精要，尽量不适用复杂华丽的修饰，要求以白话形式描述，而非文学语言。
并且要求答案有依据可查，不得包含源文本外的虚构内容，允许部分润色，但不允许造假。
要求回复必须以严格JSON格式输出，不得包含其他内容。
示例：
[
{{"content": "我依赖对方获取安全感，但这种依赖导致他人反感，让我自我怀疑。我渴望被认可却不确定自身价值，反复在情感依赖与自我否定间挣扎，既想承认自私又害怕被否定，这种矛盾让我在悔恨与行动中反复循环。"}}
]
问题：{context}
本次目标文本：
{str(result_Self | result_Mem)}
'''
        # set 直接相加是非法操作，应该用并集运算
        answer = llm_ask(promote, mode='high')

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Mem-Compression Failed.')
            return 'Error'
        # return data[0]['content']
        # 这部分没有更好。
        promote = f'''
你是一个信息处理机器，请用给出文本对问题给予回复(日常聊天对话也被视为问题)。
当给出文本的内容不足以支撑对问题的回复时，允许自由发挥。
一切目标以“人类回复”为主，不必应用给出的全部文本信息，其仅作为“我”这个人对相关内容的思考和认识的参考。
注意，根据问题严肃程度，请自行决定透露信息量和输出文本长度。不建议贸然对短文本给予大量信息和长回复。
注意，自由发挥部分不得严重偏离“我”可能的基本形象。即，不得上下矛盾，包括但不限于性格矛盾、逻辑矛盾等。
特别注意，你的目标是以人类的态度回答问题，重点在于回答问题，而非毫无保留的提供一切数据。你的回复必须切题。
要求回复必须以严格JSON格式输出，不得包含其他内容。
示例：
Q:你会控制你喜欢的人吗？你觉得你是一个很自私的人吗，未来满足去控制对方？
A:
[
{{"content": "我有时候会依赖对方，但更多是想要安全感，不是想去控制。也许看起来有点自私，但我并不想剥夺对方的自由。"}}
]
本次问题：{context}
本次任务文本：
{data[0]['content']}
'''
        answer = llm_ask(promote, mode='high')

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Humanization Failed.')
            return 'Error'
        return data[0]['content']


    def selfModeling(self, context, me=None):
        # 注意：context表示上下文，content表示回复内容
        '''
        自我建模：
            1.根据一组对话对其产生问题
            2.分别回答这些问题
            3.写入向量数据库
        '''
        # 根据一组对话对其产生问题
        promote = f'''
你是一个自我建模机器，请对如下文字做分析，对文本提出多条关于“我”的问题。
若输入文本为常规文本材料，默认文本撰写者即“我”；若为连续对话类材料，默认“{'Alice' if me == None else me}”为“我”（自行根据上下文推断指代）
若提供了发言者，但却只有一条信息，默认按照书信理解。
自行按内容划分不同提问维度，每个维度只取最重要的问题。且问题数不得超过5条，并以严格的json输出（只输出json，不得输出其他内容）。
示例：
[
{{"content": "你最想/最害怕在自我认知上发现或承认的，究竟是什么？"}},
{{"content": "你的这个“最高准则”具体包含哪些核心信念或禁忌，它如何指导你在感情与生活中的决断？"}},
{{"content": "在爱与占有之间，你如何衡量保护对方和尊重对方自由的界限？"}}
]
本次任务文本：
{context}
'''
        answer = llm_ask(promote)
        # 回答问题
        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Self-Modeling Question Failed.')
            return
        cognition = ''
        for item in data:
            if(not 'content' in item):
                print('Error: Self-Modeling Answer Failed.')
                return
            promote = f'''
请使用下面的文本回复该问题：{item['content']}
要求以描述性的语句回答，简短精要，尽量不适用复杂华丽的修饰，要求以白话形式描述，而非文学语言。
每条context必须具备完备性，其自身就具备所有必须内容，不需要其他上下文补充解释。
并且，每条context必须包含“我”，如"我认为"、"我最想发现的是"等。
并且，根据所提供的数据决定事件重要程度（重要程度从1到10）。
特别注意，不可使用"你""他们"这样的抽象代词，必须使用具体指代如姓名、组织名、具体关系名（请自行根据上下文推断指代，若无法推断则使用“<未知>”代替）。
回复应该包含问题的关键内容,并以严格的json输出（只输出json，不得输出其他内容）。
示例：
问题：你最想/最害怕在自我认知上发现或承认的，究竟是什么？
回复：
[
{{"content": "在自我认知上我最想发现的是：自己的内心一直在翻腾，不会满足、会保持追求。", "weight": 8}},
{{"content": "在自我认知上我最害怕承认的是：我其实很自私，或者有一天变得心安理得、沉默不前。", "weight": 8}}
]
本次任务文本：
{context}
'''
            answer = llm_ask(promote)
            
            data2 = json.loads(answer.strip())
            if(len(data2) == 0):
                print('Error: Self-Modeling Answer Failed.')
                return
        
            for item in data2:
                if(not 'content' in item or not 'weight' in item):
                    return
                cognition += str(item) + "\n"
                self.SelfDB.add(llm_embedding(item['content']), item)
                
        # 事件记忆建模
        promote = f'''
你是一个事件记忆建模机器，请对如下文字做分析，分条找出对文本叙述人来说重要事件以及琐碎事件。
以客观的语言(使用第一人称代词“我”)和角度表述客观存在的事件(如果是“我认为”而非客观事实，则必须加以修饰)。
并且，根据所提供的数据决定事件重要程度（重要程度从1到10）。
要求以描述性的语句回答，简短精要，尽量不适用复杂华丽的修饰，要求以白话形式描述，而非文学语言。
每条context必须具备完备性，其自身就具备所有必须内容，不需要其他上下文补充解释。
若输入文本为常规文本材料，默认文本撰写者即“我”；若为连续对话类材料，默认“{'Alice' if me == None else me}”为“我”（自行根据上下文推断指代）
若提供了发言者，但却只有一条信息，默认按照书信理解。
注意：
特别注意，不可使用"你""他们"这样的抽象代词，必须使用具体指代如姓名、组织名、具体关系名（请自行根据上下文推断指代，若无法推断则使用“<未知>”代替）。
特别注意，输出内容不必包含预先提供数据，其为“我”的内在思考，只作为辅助判断的参考数据使用。
要求回复必须以严格JSON格式输出，不得包含其他内容。
示例：
[
{{"content": "昨天我写了一封卑鄙的信，并认为银河因此伤心，我为此感到后悔并承认错误。", "weight": 9}},
{{"content": "我强调自己“还不算太笨、还能干很多事”，这是我在表述依赖与求助时保留的自我能力认知。", "weight": 5}}
]
预先提供数据：
{cognition}
本次任务文本：
{context}
'''
        answer = llm_ask(promote)

        data = json.loads(answer.strip())
        if(len(data) == 0):
            print('Error: Memory-Modeling Answer Failed.')
            return

        for item in data:
            if(not 'content' in item or not 'weight' in item):
                return
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