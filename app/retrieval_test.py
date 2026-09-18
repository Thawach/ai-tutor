from app.services.knowledge_service import (
    KnowledgeService,
)


TEST_QUERIES = [
    "ทรานซิสเตอร์ NPN มีโครงสร้างอย่างไร",
    "ทรานซิสเตอร์ PNP มีโครงสร้างอย่างไร",
    "ขาของทรานซิสเตอร์ BJT มีอะไรบ้าง",
    "กระแสของ NPN และ PNP ไหลต่างกันอย่างไร",
    "Base Current และ Collector Current สัมพันธ์กันอย่างไร",
]


def main():

    service = KnowledgeService()

    for query in TEST_QUERIES:

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        result = service.retrieve(
            query=query,
            n_results=5,
        )

        for index, source in enumerate(
            result.sources,
            start=1,
        ):

            print(
                f"{index}. "
                f"{source.source} "
                f"| Page={source.page} "
                f"| Distance={source.distance}"
            )


if __name__ == "__main__":
    main()