INSTRUCTIONS = '''
Your task is to answer questions from the course llm-zoomcamp
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model
        self.total_tokens = 0

    def search(self, query, num_results=5):
        return self.index.search(
            query,
            num_results=num_results,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['content'])
            lines.append('')

        context = '\n'.join(lines).strip()

        return context

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        print(context)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        usage = response.usage
        self.total_tokens = usage.input_tokens + usage.output_tokens
        
        return response.output_text
    
  
    def get_usage(self):
        if self.total_tokens is None:
            return "No usage data available. Run rag() first."

        return self.total_tokens
    
    def rag(self, query):
        search_results = self.search(query)
        print(search_results)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)

        
        return answer