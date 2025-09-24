import json
from sklearn.cluster import KMeans
from module.LlamaRequest import llm_ask, llm_embedding
from module.NLPNPrompts import *
from module.VectorDB import SimpleVectorDB
def llm(message, mode='low'):
    # LLM封装
    return llm_ask(message, mode)

class NLPN:
    '''
    Nature Language Processing Network
    通过NLP算子完成信息压缩，生成高维信息
    
    特点：
    1.K-Means聚类数量依据每次传入Data数决定，DataNumber/K = 当前NLP算子有效处理数据量
        可以按字数计算，但这里用条数实现即可
    2.传出数据数根据传入数据数决定，
        当 (HiddenOutput < 0.6 * Input) 时，结束隐藏层计算并传入输出处理层即后处理层
    注：由于该网络算力消耗极高，所以建议只有当已有数据量达到一定阈值时再触发

    或许每个NLP算子可以同时处理多条数据，以节省算力，
        不然每条都调用NLP算子，真的很难想象其所需算力级别。
    不过，我相信，随着技术的发展LLM的算力成本将会无比廉价。
    大语言模型的本质就是不断提高其重要信息召回能力以实现对信息的高度抽象，
        并基于抽象数据生成结果。而随着LLM的发展，其为自然语言处理提供了新的范式，
        在此，我选择将LLM作为中间层，以测试将信息抽象上升一个维度会是怎样的结果。
    需要注意的是，这里不是无意义的信息压缩，而是使用数量换取密度信息压缩。
    其在少量信息的情况下一定是无效甚至有负面影响的。
    '''
    def __init__(self):
        pass

    def inputLayer(self, AuxiliaryData, SourceData):
        '''
        目的：完成预处理实现辅助数据与源数据混合，以便更好完成任务
        作用：
            完成AuxiliaryData与SourceData的带标签聚类(类型标签)
            (聚类完毕后由外部调用将其传入HiddenLayer)
        示例：
            在认知系统中，AuxiliaryData即为记忆数据，SourceData为初级认知数据
            记忆与认知是不可分的，可以一次性传入全部可能有关联的数据，即相同时间步产生的数据
        
        AuxiliaryData -> (text-list, embedding-list) -> Tag 0
        SourceData -> (text-list, embedding-list)    -> Tag 1
        
        '''
        # 完成数据组合，以便完成K-Means
        # [[text], [embedding], [tag]]
        mixData = [
            AuxiliaryData[0] + SourceData[0],
            AuxiliaryData[1] + SourceData[1],
            [0] * len(AuxiliaryData[0]) + [1] * len(SourceData[0])
        ]
        # 确定聚类数量，每类约15个数据
        numClusters = int(len(mixData[0]) / 15)
        if numClusters == 0:
            # 这里可以优化，但在逻辑正常的情况下不会进入该过程，则就这样吧
            numClusters = 1
        # K-Means
        kmeans = KMeans(n_clusters=numClusters, random_state=42, n_init=12)
        kmeans.fit(mixData[1])
        # 聚类数据分类
        # classifiedData[label] -> [(text, type), ...]
        # 这里预留的embedding传递的可能
        classifiedData = [[] for _ in range(numClusters)]
        for idx, label in enumerate(kmeans.labels_):
            classifiedData[label] += [(mixData[0][idx], mixData[2][idx])]
        return classifiedData
    
    def hiddenLayer(self, classifiedData, aimNum):
        '''
        输入：aimNue为-1时不进行结束检验，为0时直接输出output
        目的：完成一级信息压缩
        作用：
            通过NLP算子(LLM)，完成一级信息压缩，并实现聚类
            (聚类完毕后由外部调用将其传入下一层HiddenLayer或OutputLayer)
        返回值：当衰减完毕时 [text, ...] | 当未衰减完毕时 classifiedData[idx] -> (text, type)
        这里的核心其实是prompt的编写
        '''
        # data -> [(text, type), ...]
        # LLM：
        # [
        # {{"content": "输出1"}},
        # {{"content": "输出2"}}
        # ]
        output = []
        for data in classifiedData:
            response = llm_ask(pmt_hiddenLayer.format(context=str(data)),mode='high')
            tempData = json.loads(response)
            if(len(tempData) != 0):
                output += [d['content'] for d in tempData]

        # 判定数据量是否衰减到目标范围内
        # aimNue为-1时不进行结束检验，为0时直接输出output
        if((len(output) <= aimNum and aimNum != -1) or aimNum == 0):
            # 区分结束输出与中间输出可判断第一项数据类型
            return output

        embeddings = []
        for text in output:
            embeddings.append(llm_embedding(text))

        # 确定聚类数量，每类约15个数据
        numClusters = int(len(output) / 15)
        if numClusters == 0:
            # 这里可以优化，但在逻辑正常的情况下不会进入该过程，则就这样吧
            numClusters = 1
        # K-Means
        kmeans = KMeans(n_clusters=numClusters, random_state=42, n_init=12)
        kmeans.fit(embeddings)
        # 聚类数据分类
        # classifiedData[label] -> [(text, type), ...]
        classifiedData = [[] for _ in range(numClusters)]
        for idx, label in enumerate(kmeans.labels_):
            # tag-1 -> SourceData
            classifiedData[label] += [(output[idx], 1)]
        return classifiedData
            
        

    def outputLayer(self, hiddenData, vectorDB):
        '''
        输入：hiddenData -> [text, ...]
        目的：
        作用：
            1.对同级高维数据库进行向量检索(已有记录数>0 时)，为每条数据寻找Top-3
                注：这里没必要对数据库已有数量做出限制，
                    毕竟对于对话或者小说这种时序数据来说，无论多细微的数据都是有关联的
            2.(本次数据 + Top-3)实现数据互补及更新，即以本次数据为主对原始数据进行更新。
                并由NLP算子决定是否增加记录以及是否删改原始记录，构造操作表
            3.本次数据进行互相Top-k，同样向量查询Top-3进行数据互补(同次同级数据互补没必要)，然后依据操作表删改数据
        注：这里已经是高维数据了，没有必要再次使用K-Means聚类。
        '''
        # 搜寻Top-k并聚类，传入hiddenLayer复用
        topK = 3
        # 注意：这里的classifiedData[idx]为了完成记录修改，其定义与其他函数中不同
        # classifiedData[idx] -> [(text, tag, id)]
        classifiedData = [[] for _ in range(len(hiddenData))]
        for idx, text in enumerate(hiddenData):
            # 要求vectorDB数据以{'content': '', ...}格式存储
            x = vectorDB.query(llm_embedding(text), topK)
            classifiedData[idx] += [(text, 1, 0)] + [(i[0]['content'], 0, i[1]) for i in x]
        # 目前Top-k聚类完毕，接下来需要完成数据融合以及对删改的实现
        for data in classifiedData:
            response = llm_ask(pmt_outputLayer.format(context=str(data)),mode='high')
            tempData = json.loads(response)
            if(len(tempData) != 0):
                return
            for d in tempData:
                # {"operation": 0, "content": "增加记录", id: 0}
                match d['operation']:
                    case 0:
                        # 增加记录
                        vectorDB.add(llm_embedding(d['content']), {'content': d['content']})
                    case 1:
                        # 删除记录
                        vectorDB.remove(d['id'])
        return

    def postLayer(self, data):
        # 该架构中，postLayer已经无用。
        pass

# core = NLPN()
# # 7条
# aux = [
#     "哈维因用凶白兽皮制作兔头帽，将炉心红宝石镶嵌成兔子眼睛，使帽子具备持续供暖功能",
#     "伊芙特罗娜在雪地里首次自主行动，包括眨眼、揉眼睛、拔腿、跳跃等肢体反应",
#     "哈维因在雪山岩壁上用火咒制造人工光源，通过控制火光确认自身存在",
#     "伊芙特罗娜摸查身体确认存在感，通过触觉感知自身形态",
#     "伊芙特罗娜在哈维因提示下说出名字'伊芙特罗娜'，确认身份记忆",
#     "哈维因将凶白兽皮进行鞣制处理，包括刮脂、清洁、涂抹药剂和烘烤工序",
#     "哈维因将凶白兽肉分割处理并制作成冻肉堆，完成基础生存物资储备",
#     "哈维因用兽尾制作围脖并用红丝带束发，完成少女基础装扮",
#     "哈维因在深夜处理凶白兽时，通过分割兽体获得可食用部分和制作材料",
#     "哈维因用登山镐在岩壁上搭建防风底座并堆砌积雪隔热层"
# ]
# source = [
#     "我认为当哈维因成为我记忆中的关键时，他的存在让我意识到自己并非孤立，而是与他共同经历过的事件相连。",
#     "我最想发现的是：哈维因的出现如何让我重新定义‘存在’——他既是救赎者也是记忆的锚点，让破碎的自我在具体行动中重新凝聚。"
# ]
# embeddings1 = []
# embeddings2 = []

# for doc in aux:
#     embeddings1.append(llm_embedding(doc))

# for doc in source:
#     embeddings2.append(llm_embedding(doc))

# db = SimpleVectorDB(1024, 10000, "./db/SelfModeling_VectorDB")
# mid = core.inputLayer((aux, embeddings1), (source, embeddings2))
# mid = core.hiddenLayer(mid, 0)
# core.outputLayer(mid, db)
# print(mid)