curl --location 'http://localhost:8080/v1/chat/completions' \
--header 'sec-ch-ua-platform: "macOS"' \
--header 'Referer: http://localhost:8080/' \
--header 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36' \
--header 'sec-ch-ua: "Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"' \
--header 'Content-Type: application/json' \
--header 'sec-ch-ua-mobile: ?0' \
--data '{
    "messages": [
        {
            "role": "user",
            "content": "你好"
        }
    ],
    "stream": false,
    "cache_prompt": true,
    "reasoning_format": "none",
    "samplers": ["penalties","dry","top_n_sigma","top_k","typ_p","top_p","min_p","xtc","temperature"],
    "temperature": 1,
    "dynatemp_range": 0,
    "dynatemp_exponent": 1,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "typical_p": 1,
    "xtc_probability": 0,
    "xtc_threshold": 0.1,
    "repeat_last_n": 64,
    "repeat_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "dry_multiplier": 0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": 2048,
    "max_tokens": 5,
    "timings_per_token": false
}'
