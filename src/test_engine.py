from updater import update_reference_lists
from credibility_engine import evaluate_source

update_reference_lists()

url = "https://www.bbc.com/news"

result = evaluate_source(url)

print(result)