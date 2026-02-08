---
layout: page
title: Reading
---

<svg style="position:absolute;width:0;height:0;visibility:hidden">
  <defs>
    <filter id="paper" x="0%" y="0%" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="8" result="noise"/>
      <feDiffuseLighting in="noise" lighting-color="white" surfaceScale="1" result="diffLight">
        <feDistantLight azimuth="45" elevation="35"/>
      </feDiffuseLighting>
    </filter>
  </defs>
</svg>

<div id="bookshelf"></div>
<div id="book-detail"></div>

<style>
#bookshelf, #bookshelf *, #book-detail, #book-detail * {
  background-color: transparent !important;
  box-sizing: border-box !important;
}

#bookshelf {
  display: flex !important;
  align-items: flex-end !important;
  gap: 12px !important;
  overflow: visible !important;
  padding: 24px 0 16px 0 !important;
  min-height: 260px !important;
}

.book-btn {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  justify-content: flex-start !important;
  outline: none !important;
  flex-shrink: 0 !important;
  perspective: 1000px !important;
  -webkit-perspective: 1000px !important;
  gap: 0 !important;
  transition: all 500ms ease !important;
  cursor: pointer !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

.book-spine {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: flex-start !important;
  width: 42px !important;
  height: 220px !important;
  flex-shrink: 0 !important;
  transform-origin: right !important;
  transform: rotateY(0deg) !important;
  transition: all 500ms ease !important;
  transform-style: preserve-3d !important;
  position: relative !important;
  overflow: hidden !important;
  filter: brightness(0.8) contrast(2) !important;
}

.book-spine.open {
  transform: rotateY(-60deg) !important;
}

.book-spine h2 {
  margin-top: 12px !important;
  font-family: "DM Sans", system-ui, -apple-system, sans-serif !important;
  font-size: 1.05rem !important;
  font-weight: 500 !important;
  writing-mode: vertical-rl !important;
  user-select: none !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  max-height: 196px !important;
  position: relative !important;
  z-index: 3 !important;
  background: transparent !important;
}

.book-spine .paper-tex {
  pointer-events: none !important;
  position: absolute !important;
  top: 0 !important;
  left: 0 !important;
  z-index: 50 !important;
  height: 220px !important;
  width: 42px !important;
  opacity: 0.4 !important;
  filter: url(#paper) !important;
}

.book-cover {
  position: relative !important;
  flex-shrink: 0 !important;
  overflow: hidden !important;
  transform-origin: left !important;
  transform: rotateY(88.8deg) !important;
  transition: all 500ms ease !important;
  transform-style: preserve-3d !important;
  height: 220px !important;
  filter: brightness(0.8) contrast(2) !important;
}

.book-cover.open {
  transform: rotateY(30deg) !important;
}

.book-cover img {
  width: 166px !important;
  height: 220px !important;
  display: block !important;
  margin: 0 !important;
  max-width: none !important;
  transition: all 500ms ease !important;
}

.book-cover .paper-tex {
  pointer-events: none !important;
  position: absolute !important;
  top: 0 !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  z-index: 50 !important;
  opacity: 0.4 !important;
  filter: url(#paper) !important;
}

.book-cover .crease {
  pointer-events: none !important;
  position: absolute !important;
  top: 0 !important;
  left: 0 !important;
  bottom: 0 !important;
  z-index: 50 !important;
  width: 166px !important;
  height: 220px !important;
  background: linear-gradient(to right, rgba(255,255,255,0) 2px, rgba(255,255,255,0.5) 3px, rgba(255,255,255,0.25) 4px, rgba(255,255,255,0.25) 6px, transparent 7px, transparent 9px, rgba(255,255,255,0.25) 9px, transparent 12px) !important;
}

#book-detail {
  max-height: 0 !important;
  overflow: hidden !important;
  opacity: 0 !important;
  transition: max-height 0.5s ease, opacity 0.4s ease, margin 0.4s ease !important;
  margin-top: 0 !important;
}

#book-detail.show {
  max-height: 60vh !important;
  overflow-y: auto !important;
  opacity: 1 !important;
  margin-top: 24px !important;
  padding-top: 24px !important;
  border-top: 1px solid #e5e5e5 !important;
}

.detail-info h2 {
  margin: 0 0 4px 0 !important;
  font-size: 1.5rem !important;
  font-weight: 700 !important;
  color: #111 !important;
  line-height: 1.3 !important;
}

.detail-info .author {
  color: #888 !important;
  font-size: 0.92rem !important;
  margin: 0 0 16px 0 !important;
}

.detail-info .desc {
  color: #444 !important;
  font-size: 1.1rem !important;
  line-height: 1.8 !important;
  margin: 0 !important;
}
</style>

<script>
(function() {
  var books = [
    {
      title: "Why We Sleep",
      author: "Matthew Walker",
      isbn: "9780141983776",
      spineColor: "#eae8e0",
      textColor: "#1a1a1a",
      desc: "this book by matthew walker, a berkeley sleep scientist, completely changed how i think about sleep. it's not just some optional thing we do - it's super important for staying healthy and literally not dying early. if you sleep less than 6 hours, you're basically screwed with way higher risks of cancer and alzheimer's, plus your brain works like you're drunk. the book explains how the first half of sleep stores facts while the second half connects ideas and helps creativity. walker says we need 7-9 hours and should avoid screens before bed, keep rooms cool, stop caffeine after noon, and stay consistent with sleep times even on weekends. some people say he exaggerates stuff a bit, but honestly it made me realize sleep matters way more than i ever thought."
    },
    {
      title: "Artificial Intelligence",
      author: "Melanie Mitchell",
      isbn: "9780241404836",
      spineColor: "#86b8ca",
      textColor: "#1a1a1a",
      desc: "melanie mitchell is this computer scientist who teaches at portland state and does research at the santa fe institute. in this book, she breaks down how ai actually works versus all the crazy hype we hear about it. she digs into machine learning, neural nets, deep learning and all that, but keeps coming back to this idea that current ai is just really good pattern matching without any real understanding. like it can do impressive stuff but has no clue what it's actually doing. she calls this the barrier of meaning - where ai can't really understand context or common sense or make the kind of connections humans easily do. she points out how brittle these systems are and how they can fail in super weird ways. the book was written before chatgpt but it's still super relevant and honestly makes complex stuff easy to get without dumbing it down too much."
    }
  ];

  function getCover(isbn) {
    return "https://covers.openlibrary.org/b/isbn/" + isbn + "-L.jpg";
  }

  var shelf = document.getElementById("bookshelf");
  var detail = document.getElementById("book-detail");
  var currentBook = -1;

  function createShelf() {
    shelf.innerHTML = "";
    for (var i = 0; i < books.length; i++) {
      (function(index) {
        var book = books[index];
        var isOpen = index === currentBook;

        var btn = document.createElement("button");
        btn.className = "book-btn";
        btn.style.cssText = "width:" + (isOpen ? "208px" : "42px") + " !important;";

        var spine = document.createElement("div");
        spine.className = "book-spine" + (isOpen ? " open" : "");
        spine.style.cssText = "background-color:" + book.spineColor + " !important;color:" + book.textColor + " !important;";

        var spineTitle = document.createElement("h2");
        spineTitle.textContent = book.title;
        spine.appendChild(spineTitle);

        var paperTex1 = document.createElement("span");
        paperTex1.className = "paper-tex";
        spine.appendChild(paperTex1);

        var cover = document.createElement("div");
        cover.className = "book-cover" + (isOpen ? " open" : "");

        var coverImg = document.createElement("img");
        coverImg.src = getCover(book.isbn);
        coverImg.alt = book.title;
        cover.appendChild(coverImg);

        var paperTex2 = document.createElement("span");
        paperTex2.className = "paper-tex";
        cover.appendChild(paperTex2);

        var crease = document.createElement("span");
        crease.className = "crease";
        cover.appendChild(crease);

        btn.appendChild(spine);
        btn.appendChild(cover);

        btn.onclick = function() {
          if (currentBook === index) {
            currentBook = -1;
            detail.classList.remove("show");
          } else {
            currentBook = index;
            showDetail(book);
          }
          createShelf();
        };

        shelf.appendChild(btn);
      })(i);
    }
  }

  function showDetail(book) {
    detail.innerHTML = '<div class="detail-info"><h2>' + book.title + '</h2><p class="author">By: ' + book.author + '</p><p class="desc">' + book.desc + '</p></div>';
    detail.classList.remove("show");
    setTimeout(function() { detail.classList.add("show"); }, 10);
  }

  createShelf();
})();
</script>
