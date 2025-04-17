from serpapi import GoogleSearch

search = GoogleSearch({
    "q": "stuttering exercises",
    "api_key": "your-serpapi-key"
})
results = search.get_dict()
print(results["organic_results"][0]["title"])