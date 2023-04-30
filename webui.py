import time
import logging
import gradio as gr

LOG = logging.getLogger(__name__)

class WebApp(object):
    def __init__(self, rds, chain, args):
        self.rds = rds
        self.chain = chain
        self.args = args

    def conversation(self, query):
        query = query.strip()
        docs = self.rds.similarity_search(query, k=3)
        LOG.info(query)
        LOG.info(docs)
        result = self.chain.run(input_documents=docs, question=query)

        return result.strip()

    def respond(self, message, chat_history):
        bot_message = self.conversation(message)
        chat_history.append((message, bot_message))
        time.sleep(1)

        return "", chat_history

    def run(self):

        with gr.Blocks() as app:
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=1000)

            with gr.Row():
                with gr.Column(scale=0.85):
                    msg = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your question"
                    ).style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("Clear")

            msg.submit(self.respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        app.launch(share=self.args.share)
