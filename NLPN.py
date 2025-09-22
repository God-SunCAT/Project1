from sklearn.cluster import KMeans
from module.LlamaRequest import llm_ask
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
        # 聚类数据分类
        # [(text, embedding, type)]
        classifiedData = [[] * numClusters]
        for idx, label in enumerate(kmeans.labels_):
            classifiedData[label] += [(mixData[0][idx], mixData[1][idx], mixData[2][idx])]
        return classifiedData
    
    def hiddenLayer(self, data):
        '''
        目的：完成一级信息压缩
        作用：
            通过NLP算子(LLM)，完成一级信息压缩，并实现聚类
            (聚类完毕后由外部调用将其传入下一层HiddenLayer或OutputLayer)
        '''
        pass

    def outputLayer(self, sameLevelData, data):
        '''
        目的：
        作用：
            1.对同级高维数据库进行向量检索(已有记录数>0 时)，为每条数据寻找Top-3
                注：这里没必要对数据库已有数量做出限制，
                    毕竟对于对话或者小说这种时序数据来说，无论多细微的数据都是有关联的
            2.(本次数据 + Top-3)实现数据互补及更新，即以本次数据为主对原始数据进行更新。
                并由NLP算子决定是否增加记录以及是否删改原始记录，构造操作表
            3.本次数据进行互相Top-k，同样向量查询Top-3进行数据互补，然后依据操作表删改数据
        注：这里已经是高维数据了，没有必要再次使用K-Means聚类。
        '''
        pass

    def postLayer(self, data):
        # 该架构中，postLayer已经无用。
        pass

core = NLPN()
core.inputLayer((['ABC'], [[1,2,3,4]]), (['def'], [[1,2,2,4]]))