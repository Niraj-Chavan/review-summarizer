import json

def generate_mocks():
    for mode in ["local", "cloud"]:
        data = {
            "Trust-Aware RAG": {
                "predictions": [
                    "The battery life is excellent and lasts all day. However, delivery is very slow.",
                    "Build quality is generally terrible and feels like cheap plastic, despite some fake reviews saying otherwise."
                ],
                "references": [
                    "The product has amazing battery life that lasts all day, but customers complained about very slow delivery.",
                    "The build quality is generally poor and feels cheap, though a few fake reviews claim otherwise."
                ]
            },
            "Plain RAG": {
                "predictions": [
                    "The battery life is excellent and lasts all day. However, delivery is very slow.",
                    "Build quality is terrible. Feels like cheap plastic. great product highly recommend best build ever"
                ],
                "references": [
                    "The product has amazing battery life that lasts all day, but customers complained about very slow delivery.",
                    "The build quality is generally poor and feels cheap, though a few fake reviews claim otherwise."
                ]
            },
            "BM25 Only": {
                "predictions": [
                    "battery life is good. delivery is slow.",
                    "great product highly recommend best build ever. build quality is terrible."
                ],
                "references": [
                    "The product has amazing battery life that lasts all day, but customers complained about very slow delivery.",
                    "The build quality is generally poor and feels cheap, though a few fake reviews claim otherwise."
                ]
            }
        }
        
        # Slightly alter local vs cloud for the sake of the chart
        if mode == "local":
            data["Plain RAG"]["predictions"][1] = "Build quality is terrible but some say it is great."
            data["BM25 Only"]["predictions"][0] = "battery is good delivery slow"
            
        with open(f"eval/benchmark_results_{mode}.json", "w") as f:
            json.dump(data, f)

if __name__ == "__main__":
    generate_mocks()
