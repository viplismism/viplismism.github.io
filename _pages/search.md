---
layout: page
---

<style>
	#search-container {
	    max-width: 100%;
	}

	input[type=text] {
		font-size: normal;
	    outline: none;
	    padding: 0.5rem;
		border-bottom: 1px solid #000;
margin-left: 0;
background: transparent;
	    width: 100%;
		-webkit-appearance: none;
		font-family: inherit;
		font-size: 100%;
		border: none;
	}
	#results-container {
		margin: 0rem 0;
	}
</style>

<script>
document.getElementById('search-input').addEventListener('input', function() {
    document.getElementById('search-line').style.display = this.value ? 'none' : 'block';
});
</script>

<!-- Html Elements for Search -->
<div id="search-container">
<input type="text" id="search-input" placeholder="">
<ol id="results-container"></ol>
</div>
<hr style="margin-top: 1rem;" id="search-line">

<style>
#search-container:focus-within #search-line {
    display: none;
}
</style>

<!-- Script pointing to search-script.js -->
<script src="/search.js" type="text/javascript"></script>

<!-- Configuration -->
<script type="text/javascript">
SimpleJekyllSearch({
  searchInput: document.getElementById('search-input'),
  resultsContainer: document.getElementById('results-container'),
  json: '/search.json',
  searchResultTemplate: '<li><a href="{url}" title="{description}">{title}</a></li>',
  noResultsText: 'No results found',
  limit: 10,
  fuzzy: false,
  exclude: ['Welcome']
})
</script>
