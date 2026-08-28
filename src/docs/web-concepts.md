## BeautifulSoup
Beautiful Soup is a Python library used for parsing HTML and XML, basically for pulling data out of web pages.
When you scrape a webpage, you usually get back this huge messy blob of raw HTML text. Beautiful Soup takes that and turns it into a structured, navigable object, so instead of manually searching through text with regex or string matching, you can do things like "find me the first paragraph tag" or "grab every link on this page" using clean, readable code.
A typical use case, you fetch a page's HTML using something like the requests library, then hand that HTML off to Beautiful Soup, and it lets you search by tag name, class, ID, or even CSS selectors, and pull out exactly the piece of content you need, like a headline, a price, or a table of data.
It's one of the most commonly used tools for web scraping in Python, partly because it's forgiving. Even if a page's HTML is a little broken or messy, which happens constantly on real websites, Beautiful Soup usually still manages to parse it without crashing.

## DOM tree (Document Object Model)
At the very top, the root, you've got the HTML tag for the whole document. That branches down into things like the head and the body. The body then branches further into elements like headers, paragraphs, divs, links, images, all nested inside each other based on how the page is actually built.
So when a browser loads a webpage, it doesn't just see one flat blob of text, it builds this tree structure in memory, where every HTML element is a node, and elements nested inside other elements become child nodes. This is what lets JavaScript, or tools like Beautiful Soup, navigate around and say "give me the third paragraph inside this specific div," because they're essentially walking through branches of that tree to find exactly the node they want.

## Requests is not a browser
requests.get(url) does exactly one thing: sends a single HTTP GET and hands you back the raw response body — the bytes the server produced. Nothing else runs.

A browser does much more after receiving that same HTML:

Parses it into a DOM tree.
Downloads every `<script>` it references.
Executes that JavaScript.
The JS often makes more HTTP calls (fetch/XHR) to backend APIs, gets JSON back, and injects new DOM nodes — headings, paragraphs, the actual article.
Only after all that is the page "complete."
Steps 2–5 are what requests skips entirely.

## Where does the content live? SSR vs CSR
Server-side rendered (SSR) / static pages — the server assembles the full HTML, text and all, before sending it. requests sees 100% of the content. (Classic CMSs: WordPress, Drupal, plain HTML.)

Client-side rendered (CSR) / single-page apps (SPA) — the server sends a near-empty skeleton:

```bash
<body>
  <header>…nav…</header>
  <div id="root"></div>          <!-- empty! -->
  <script src="/app.bundle.js"></script>
</body>
```

The `<div id="root">` gets filled in by JavaScript after load — that filling-in is called hydration. requests receives only the skeleton. The header/footer/nav are usually in the static HTML (for SEO and fast first paint), but the body content isn't — that's the "nav shell" I meant: you get the frame around the content but not the content. Then clean_html strips nav/header/footer, and you're left with almost nothing.

Frameworks that commonly do this: React, Vue, Angular, Svelte SPAs (look for `<div id="__next">, <div id="root">, <app-root>`).