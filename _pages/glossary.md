---
layout: page
title: Glossary
---
<div class="glossary-index">
  <ul>
    {% assign entries = site.glossary | sort: "title" %}
    {% for entry in entries %}
      <li><a href="{{ entry.url | relative_url }}">{{ entry.title }}</a></li>
    {% endfor %}
  </ul>
</div>
