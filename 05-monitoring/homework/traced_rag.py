from rag_helper import RAGBase

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class RAGTraced(RAGBase):

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:

            return super().search(
                query,
                num_results=num_results
            )

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)

            span.set_attribute(
                "input_tokens",
                response.usage.input_tokens
            )
            span.set_attribute(
                "output_tokens",
                response.usage.output_tokens
            )
            span.set_attribute(
                "total_tokens",
                response.usage.total_tokens
            )

            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:

            return super().rag(query)