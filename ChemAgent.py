import time
from abc import abstractmethod
from collections import Counter
from typing import Tuple, List, Optional
from retry import retry
from config import OPENAI_API_KEY, OPENAI_BASE_URL_PROXY

from langchain_openai import ChatOpenAI
from langchain.utilities import BingSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate
import utils
import prompts

#TODO LocalAgent
#TODO prompts
#TODO history or log and some utils and fix bugs
#TODO divide into humanmessage and systemmessage

class BaseAgent():
    def __init__(self, alpha: int = 0.5, state: str = "Start", vector_store: str = "langchain-chatchat") -> None:
        # Basic config
        self.question = None
        self.state_list = ["Start", "Analyse", "Retrieve", "WebSearch", "Lookup", "Finish"]
        self.state = state
        self.Webwrapper = BingSearchAPIWrapper()
        self.vector_store = vector_store
        # hyper parameter config
        self.complexity_weight = alpha
        self.accesibility_weight = 1 - alpha
        
        self._reset()
    
    def _reset(self):
        self.pre_state = "Start"
        self.answer = None
        self.Intermediate_results = list()
        #TODO RAG API if there is a interface to obtain content else RAG from scratch
        
        self.history_list = list()
    
    @retry(tries=3)
    @abstractmethod
    def _call(self, input: str) -> str:
        """
        call LLM and return its answer, need implementing by subclass

        Args:
            input (str): any input in forms of str 

        """
        raise NotImplementedError

    def _decide_next_step(self, previous_result: str) -> str:
        decide_next_step_prompt = prompts.DECIDE_NEXT_STEP_TEMPLATE.format(
            previous_result = previous_result,
            cur_state = self.state,
            example = prompts.NEXT_STEP_EXAMPLE,
        )
        
        response = self._call(decide_next_step_prompt)
        next_step = self._abstract_step_from_response(response)
        
        return next_step
        
    def _Analyse(self, pre_result: Optional[List[str]] = None) -> str:
        """
        Analyse the situation and decide what to do. Options include Retrieve, Web and Finish

        Args:
            pre_result (Optional[List[str]], optional): obtained result so far, 
                                                        None indicates the beginning. Defaults to None.

        Returns:
            str: next step decision, analysis included 
        """
        self.pre_state = self.state
        self.state = "Analyse"
        if(pre_result == None):
            analyse_prompt = prompts.ANALYSE_SCORE_TEMPLATE.format(
                question = self.question,
                example = prompts.ANALYSE_SCORE_EXAMPLE, 
                accesibility_weight = self.accesibility_weight,
                complexity_weight = self.complexity_weight,
            )
        
        else:
            analyse_prompt = prompts.ANALYSE_FINISH_TEMPLATE.format(
                question = self.question,
                example = prompts.ANALYSE_FINISH_EXAMPLE, 
                pre_result = pre_result,
            )
        
        
        response = self._call(analyse_prompt)
        
        return response

    def _Retrieve(self, num_of_knowledge: int = 1) -> List[str]:
        """
        Retrieve knowledge base and return corresponding knowledge

        Args:
            num_of_knowledge (int, optional): Amount of knowledge that needed. Defaults to 1.

        Returns:
            List[str]: List of knowledge
        """
        
        self.pre_state = self.state
        self.state = "Retrieve"
        
        def rewrite_query(question: str):
            rewrite_query_prompt = prompts.REWRIE_QUERY_TEMPLATE.format(
                query = question,
            )
            
            rewritten_query = self._call(rewrite_query_prompt)
            
            return rewritten_query
            
        query = rewrite_query(self.question)
        retrieve_result = self.access_knowledge_base(query=query,  num_of_example=num_of_knowledge)
        #TODO Format maybe required
        return retrieve_result
    
    def _Websearch(self, num_of_search: int = 3) -> List[str]:
        """
        Search through Internet and return corresponding results

        Args:
            num_of_search (int, optional): bumber of search results that needed. Defaults to 3.

        Returns:
            List[str]: List of results
        """
        
        self.pre_state = self.state
        self.state = "WebSearch"
        
        def rewrite_search(question: str):
            rewrite_search_prompt = prompts.REWRITE_SEARCH_TEMPLATE.format(
                question = question,
            )
            
            rewritten_search = self._call(rewrite_search_prompt)
            
            return rewritten_search
        
        search = rewrite_search
        search_result = self.Webwrapper.results(search, num_of_search)
        #TODO Format maybe required
        return search_result

    def _Lookup(self, pre_result: List[str]) -> str:
        """
        Lookup the intermedia result to obtain useful infomation

        Args:
            pre_result (List[str]): List of intermedia results from Retrieve or Websearch

        Returns:
            str: Useful knowledge
        """
        
        self.pre_state = self.state
        self.state = "Lookup"
        
        lookup_prompt = prompts.LOOKUP_TEMPLATE.format(
            pre_result = pre_result,
            question = self.question,
            
        )
        
        response = self._call(lookup_prompt)
        return response
    
    
    @abstractmethod
    def _answer(self, intermediate_result: List[str]) -> str:
        raise NotImplementedError

    @retry(tries=3)
    def run(self, question: str, max_steps: int = 7):
        self.question = question
        
        for num_of_step in range(max_steps):
            
            self.print_basic_info()
            if self.state == "Start":
                self.Intermediate_results.append(self._Analyse())
                
            elif self.state == "Analyse":
                next_step = self._decide_next_step(self.Intermediate_results[-1])
                if next_step == "Retrieve":
                    self.Intermediate_results += self._Retrieve()
                elif next_step == "WebSearch":
                    self.Intermediate_results += self._Websearch()
                elif next_step == "Finish":
                    break
                else:
                    print("-"*10, "😨Analyse produce invalid step😨", "-"*10)
                    raise Exception
                    
            elif self.state == "Retrieve":
                self.Intermediate_results[-1] = (self._Lookup(self.Intermediate_results[-1]))
            
            elif self.state == "WebSearch":
                self.Intermediate_results[-1] = (self._Lookup(self.Intermediate_results[-1]))
                
            elif self.state == "Lookup":
                self.Intermediate_results.append(self._Analyse())
            
            
                
        if self.state == "Finish" or (num_of_step >= max_steps):
            self.state = "Finish"
            self.answer = self._answer(self.Intermediate_results)
            return self.answer
    
    # Utils & property 

    def print_cur_state(self):
        print("==\nState info", "="*28, "\n")
        print("-"*10, f"🗣current state is {self.state}", "-"*10, "\n")
        print("-"*10, f"🗣previous state is {self.pre_state}", "-"*10, "\n")

    def print_intermedia_results(self):
        #TODO format may needed
        print("\nIntermedia results info", "="*30, "\n")
        print("-"*10, f"😑Intermedia results are {self.Intermediate_results}", "-"*10, "\n")
    
    def print_basic_info(self):
        print("\nBasic info", "="*30, "\n")
        self.print_cur_state()
        self.print_intermedia_results()
        print("\nEnd", "="*37, "\n\n")
        
    
    def _abstract_step_from_response(self, response: str):
        #TODO abstract step using re, implemented after prompts
        pass
    
    def access_knowledge_base(self, query: str, num_of_example: int = 1, threshold: float = 0.5) -> List[str]:
        if self.vector_store == "langchain-chatchat":
            knowledge = utils.kb_chat(query=query,
                                    top_k=num_of_example,
                                    score_threshold=threshold,)
        
        return knowledge
    
    @property
    def get_state(self):
        return self.state
    
    #TODO interface interact with fe
    
    
class OpenAIAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4", vector_store: str = "langchain-chatchat", state: str = "Start"):
        super().__init__(vector_store=vector_store, state=state)
        self.model = model
        self._set_llm()
    
    def _set_llm(self):
        llm = ChatOpenAI(
            model = self.model,
            temperature = 0.5,
            max_retries = 3,
        )
        
        self.llm = llm
              
    def _call(self, input: str) -> str:
        
        prompt = HumanMessagePromptTemplate.from_template("{prompt}")
        chain = prompt | self.llm | StrOutputParser
        response = chain.invoke({"prompt": input})
        
        return response
    
    def _answer(self, intermediate_result: List[str]) -> str:
        # def answer_parser(answer: str):
        #     pass
        #TODO format maybe required
        answer_prompt = prompts.ANSWER_TEMPLATE.format(
            question = self.question,
            intermediate_result = intermediate_result,
        )
        
        answer = self._call(answer_prompt)
        
        return answer

class LocalAgent(BaseAgent):
    pass

