from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset


def create_movie_test_set():
    """Movie-specific test questions with ground truth."""
    return [
        {
            "question": "What is Inception about?",
            "ground_truth": "Inception is about a thief who enters dreams to steal secrets. He's offered a chance to have his criminal history erased if he can plant an idea in someone's mind through dream inception.",
            "query_type": "factual",
        },
        {
            "question": "What makes Christopher Nolan's directing style unique?",
            "ground_truth": "Nolan is known for non-linear storytelling, complex narratives, practical effects over CGI, philosophical themes about time and memory, and intricate puzzle-like plots that reward multiple viewings.",
            "query_type": "analytical",
        },
        {
            "question": "Recommend sci-fi movies with time travel themes",
            "ground_truth": "Sci-fi movies with time travel include Primer (complex low-budget), 12 Monkeys (dystopian), Looper (action-focused), Interstellar (space-time), Back to the Future (classic), and The Terminator (action).",
            "query_type": "recommendation",
        },
        {
            "question": "What are the main themes in The Matrix?",
            "ground_truth": "The Matrix explores themes of reality vs illusion, free will vs determinism, technology's control over humanity, choosing uncomfortable truth over comfortable lies, and the hero's journey of self-discovery.",
            "query_type": "thematic",
        },
        {
            "question": "What do critics say about The Dark Knight?",
            "ground_truth": "Critics praised The Dark Knight for Heath Ledger's performance as the Joker, its dark and realistic tone, complex themes about chaos and order, and elevating the superhero genre to serious cinema.",
            "query_type": "opinion",
        },
        {
            "question": "Movies with twist endings?",
            "ground_truth": "Movies with twist endings include The Sixth Sense, Fight Club, and Shutter Island.",
            "query_type": "recommendation",
        },
        {
            "question": "What is the plot of Pulp Fiction?",
            "ground_truth": "Pulp Fiction interweaves multiple storylines: hitmen Vincent and Jules retrieving a briefcase, boxer Butch betraying a crime boss, and Vincent taking the boss's wife Mia out. The non-linear narrative connects these stories.",
            "query_type": "factual",
        },
        {
            "question": "Why is Citizen Kane considered important in film history?",
            "ground_truth": "Citizen Kane revolutionized cinema with deep focus cinematography, innovative camera angles, non-linear storytelling, complex character study, and technical innovations that influenced generations of filmmakers.",
            "query_type": "analytical",
        },
        {
            "question": "Compare Tarantino and Scorsese's directing styles",
            "ground_truth": "Tarantino is known for non-linear narratives, stylized violence, pop culture references, and sharp dialogue. Scorsese focuses on character studies, moral complexity, crime dramas, and masterful use of music and editing.",
            "query_type": "comparison",
        },
        {
            "question": "What makes Parasite unique as a film?",
            "ground_truth": "Parasite uniquely blends genres (comedy, thriller, drama), explores class inequality through a literal upper/lower house metaphor, maintains tonal shifts, and became the first foreign language film to win Best Picture at the Oscars.",
            "query_type": "analytical",
        },
        {
            "question": "Recommend coming of age movies about love, poverty, and friendships growing apart",
            "ground_truth": "Coming of age movies with these themes include Moonlight (explores identity, love, and poverty in three chapters), City of God (Brazilian favela, friendship torn by crime and poverty), Stand By Me (childhood friendships that drift apart over time), The Florida Project (childhood innocence against backdrop of poverty), The Outsiders (class divisions and friendship bonds), and Lady Bird (navigating relationships while dealing with family financial struggles). These films authentically portray how economic hardship and life circumstances can strain friendships while young people navigate first love.",
            "query_type": "recommendation",
        },
    ]


def evaluate_chain(chain, test_set):
    """Run evaluation on chain."""
    print("Running chain on test set...")

    results = []
    for i, test in enumerate(test_set, 1):
        print(f"  [{i}/{len(test_set)}] {test['question']}")

        result = chain.query(test["question"])

        print(f"answer:{result['answer'][:300]}...")

        raw_contexts = [doc["content"] for doc in result["sources"][:5]]
        print(f"contexts:{[text[:100] for text in raw_contexts]}...")

        results.append(
            {
                "question": test["question"],
                "answer": result["answer"],
                "contexts": raw_contexts,
                "ground_truth": test["ground_truth"],
                "query_type": test["query_type"],
            }
        )

    # Evaluate
    dataset = Dataset.from_list(results)
    scores = evaluate(
        dataset,
        # metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        metrics=[faithfulness, answer_relevancy],
    )
    # answer being faithful to context, answer being relevant to question,
    # retrieved context being relevant to question, retrieved context containing ground truth info

    return scores


def print_results(scores):
    """Pretty print results."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    print("\nScores:")
    print(f"  Faithfulness:      {scores['faithfulness']}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']}")
    # print(f"  Context Precision: {scores['context_precision']}")
    # print(f"  Context Recall: {scores['context_recall']}")
