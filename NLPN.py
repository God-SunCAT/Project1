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
        '''
        pass
    
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