import json

with open('pinterest.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Путь к пинам может отличаться, попробуйте разные варианты:
# Вариант 1: через resourceResponses
pins = data.get('resourceResponses', [])
if pins:
    pins = pins[0].get('response', {}).get('data', [])

# Вариант 2: через resourceData (если вариант 1 не дал результатов)
if not pins:
    pins = data.get('resourceData', {}).get('BoardFeedResource', {}).get('data', [])

urls = []
for pin in pins:
    # Ищем оригинальное изображение
    if 'images' in pin:
        for size in ['orig', '564x', '736x']:
            if size in pin['images']:
                urls.append(pin['images'][size]['url'])
                break

# Сохраняем в файл
with open('urls.txt', 'w', encoding='utf-8') as f:
    for url in urls:
        f.write(url + '\n')

print(f'Найдено {len(urls)} ссылок. Сохранено в urls.txt')