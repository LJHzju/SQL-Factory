import argparse
import io
import logging
import os
from logging import FileHandler

from dotenv import load_dotenv
from PIL import Image

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

from graph import init_graph


def show_graph(graph):
    print(graph.get_graph().draw_mermaid())
    exit(0)

    image = graph.get_graph().draw_mermaid_png()
    image = Image.open(io.BytesIO(image))

    image.save("graph.png", "PNG")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--benchmark", type=str, default="imdb")
    parser.add_argument("--num-sql", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()
    print(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            FileHandler(
                "log-" + args.benchmark + ".log",
                mode="a",
                encoding="utf-8",
                delay=False,
            )
        ],
    )

    graph = init_graph()
    # show_graph(graph)

    params = {}
    params["epoch"] = args.epoch
    params["benchmark"] = args.benchmark
    params["target_sql_num"] = args.num_sql

    for event in graph.stream(
        params,
        config={"recursion_limit": 999999999},
        stream_mode="updates",
    ):
        logging.info(event)
        logging.info("=" * 100 + "\n")
        
        # print(event)
        # print("=" * 100 + "\n")
