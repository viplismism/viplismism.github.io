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
  max-height: none !important;
  overflow: visible !important;
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
      title: "Sapiens A Brief History of Humankind",
      author: "Yuval Noah Harari",
      coverId: "8634250",
      localCover: "sapiens",
      spineColor: "#faf1ef",
      textColor: "#111111",
      desc: "yuval noah harari is this israeli historian who basically wrote the book on how we went from being just another ape to running the whole planet. the core idea that blew my mind is that humans dominate because we can cooperate in massive numbers through shared myths - things like money, nations, religions, corporations - none of these actually exist except in our collective imagination. he calls it the cognitive revolution, when we started telling stories and believing in things we can't see or touch. the agricultural revolution gets absolutely roasted - he calls it history's biggest fraud because we thought we domesticated wheat but wheat actually domesticated us, chaining us to backbreaking farm work and worse diets. he goes through empires, science, capitalism, all of it, showing how these imagined orders shape everything. some people say he oversimplifies stuff and cherry-picks evidence, which is fair, but honestly it completely changed how i think about why society works the way it does. like once you see that money is just a shared story we all believe in, you can't unsee it."
    },
    {
      title: "Why We Sleep",
      author: "Matthew Walker",
      coverId: "8814155",
      localCover: "why-we-sleep",
      spineColor: "#0b2545",
      textColor: "#f0f0f0",
      desc: "this book by matthew walker, a berkeley sleep scientist, completely changed how i think about sleep. it's not just some optional thing we do - it's super important for staying healthy and literally not dying early. if you sleep less than 6 hours, you're basically screwed with way higher risks of cancer and alzheimer's, plus your brain works like you're drunk. the book explains how the first half of sleep stores facts while the second half connects ideas and helps creativity. walker says we need 7-9 hours and should avoid screens before bed, keep rooms cool, stop caffeine after noon, and stay consistent with sleep times even on weekends. some people say he exaggerates stuff a bit, but honestly it made me realize sleep matters way more than i ever thought."
    },
    {
      title: "Artificial Intelligence A Guide for Thinking Humans",
      author: "Melanie Mitchell",
      coverId: "9333846",
      localCover: "ai-guide",
      spineColor: "#c2c2c6",
      textColor: "#111111",
      desc: "melanie mitchell is this computer scientist who teaches at portland state and does research at the santa fe institute. in this book, she breaks down how ai actually works versus all the crazy hype we hear about it. she digs into machine learning, neural nets, deep learning and all that, but keeps coming back to this idea that current ai is just really good pattern matching without any real understanding. like it can do impressive stuff but has no clue what it's actually doing. she calls this the barrier of meaning - where ai can't really understand context or common sense or make the kind of connections humans easily do. she points out how brittle these systems are and how they can fail in super weird ways. the book was written before chatgpt but it's still super relevant and honestly makes complex stuff easy to get without dumbing it down too much."
    },
    {
      title: "In Search of Lost Time",
      author: "Marcel Proust",
      coverId: "12332709",
      localCover: "in-search-of-lost-time",
      spineColor: "#513b2c",
      textColor: "#f0f0f0",
      desc: "marcel proust was this french writer who basically created the ultimate novel about memory and time - like, this thing is seven volumes and over 4,000 pages total, and the first volume alone is a masterpiece. the whole thing starts with the famous madeleine cookie scene where the narrator dips a cookie in tea and suddenly remembers his entire childhood, which launches this massive exploration of how memory works and how the past lives inside us. proust's big insight is that our real life exists in these moments when the past suddenly comes flooding back through random triggers - a smell, a taste, a sound - and time basically collapses. he calls it involuntary memory, and it's way more powerful than just trying to remember stuff on purpose. the writing is super dense and the sentences go on forever, which some people find exhausting, but honestly it's like being inside someone's mind as they're actually thinking and remembering. the book completely changed how i understand that memories aren't just stored facts but living things that reshape who we are, and that the present moment is always layered with all these invisible connections to our past."
    },
    {
      title: "Build a Large Language Model From Scratch",
      author: "Sebastian Raschka",
      coverId: "15225524",
      localCover: "build-llm",
      spineColor: "#1f0d13",
      textColor: "#f0f0f0",
      desc: "sebastian raschka is a machine learning researcher who decided to just build the whole thing from zero - and i mean from zero. no huggingface, no shortcuts, just numpy and pytorch and a lot of patience. the book walks you through every single component: how tokenization actually works at the byte level, how attention is implemented as actual matrix multiplications, how positional embeddings get baked in, how the training loop updates weights. the part that hit different for me was building the attention mechanism from scratch - you go from 'i kind of understand attention' to 'i can write every line of it myself and explain what each dimension means.' there's also a solid section on pretraining vs fine-tuning and how RLHF fits in. honestly after reading most ML books you still have this gap between the high-level idea and the actual code - this one closes that gap completely. if you're working on anything inference or post-training related, this is the book that makes everything else click."
    },
    {
      title: "The Alignment Problem",
      author: "Brian Christian",
      coverId: "10678431",
      localCover: "alignment-problem",
      spineColor: "#e4ece8",
      textColor: "#111111",
      desc: "brian christian spent years interviewing the actual researchers working on making AI systems do what we want - and the picture he paints is simultaneously fascinating and kind of terrifying. the alignment problem is basically: how do you make a machine that pursues the goals you actually have instead of a technically-correct but catastrophically-wrong version of them. he goes through real cases - reward hacking where an RL agent finds absurd shortcuts that technically maximize the reward, specification gaming where the AI does exactly what you asked but not what you meant, brittleness in vision systems that fail on examples a child would get right. the book is structured as a history of the field, so you get context on why researchers started worrying about this stuff and how the conversation evolved. it's not a doomer book and it's not naive either - it's just an honest look at how hard it is to specify human values in a form a machine can optimize for. after spending time on inference bugs and watching models produce confidently wrong output, the core thesis hit closer to home than i expected."
    },
    {
      title: "Project Hail Mary",
      author: "Andy Weir",
      coverId: "11200092",
      localCover: "project-hail-mary",
      spineColor: "#332423",
      textColor: "#f0f0f0",
      desc: "andy weir wrote the martian, and this is him doing it again but bigger and weirder. the setup: a guy wakes up alone on a spaceship with no memory of who he is or why he's there, and slowly figures out that he's been sent on a one-way mission to save the earth from an extinction-level event. it's hard sci-fi, which means the science is real and the problem-solving is the whole point - the protagonist is a scientist and he thinks like one, running experiments, forming hypotheses, updating when he's wrong. the thing that makes it work is that it never feels like a textbook - the science is just how this person thinks, the same way a good engineer would work through a production bug. there's also this completely unexpected friendship at the center of the book that lands in a way i didn't see coming at all. genuinely one of the most fun reads i've had - finished it in two sittings and felt good about it for days."
    },
    {
      title: "How to Build a Car",
      author: "Adrian Newey",
      coverId: "13168271",
      localCover: "how-to-build-a-car",
      spineColor: "#024a7a",
      textColor: "#f0f0f0",
      desc: "adrian newey is the guy who designed most of the dominant f1 cars of the last thirty years - red bull, mclaren, williams - and this is basically his autobiography told through the cars. what makes it different from a typical sports memoir is that he actually explains the engineering. like genuinely explains it - downforce, aerodynamics, suspension geometry, how changing one part of the car creates three new problems somewhere else. you realize that an f1 car is this impossibly constrained optimization problem where everything trades off against everything else, and the best designers are the ones who can hold the whole system in their head at once. the stories from the paddock are great too - the politics, the crashes, the moments where a design decision you made at the drawing board shows up at 200mph. but the thing that stayed with me is how newey thinks about problems: always going back to first principles, never accepting that something has to be the way it's always been done. honestly reads like a masterclass in engineering thinking wrapped in an f1 story."
    },
    {
      title: "The Secret",
      author: "Rhonda Byrne",
      coverId: "845815",
      localCover: "the-secret",
      spineColor: "#7d4632",
      textColor: "#f0f0f0",
      desc: "rhonda byrne's the secret is built around the law of attraction - the idea that your thoughts have a frequency, and what you focus on consistently is what you pull into your life. it's a big claim and a lot of people write it off immediately, but the core practical insight is harder to dismiss: what you think about shapes how you act, and how you act shapes what happens to you. whether that's literally magnetic or just psychology doesn't really matter when the result is the same. the book asks you to be specific about what you want, visualize it clearly, and act as if it's already on its way. the stories can get a bit much and the science it claims isn't science, but underneath all that is something real about intention and attention and how most people spend their mental energy focused on what they don't want instead of what they do. i took from it what was useful and left the rest."
    },
    {
      title: "Thinking Fast and Slow",
      author: "Daniel Kahneman",
      coverId: "13290711",
      localCover: "thinking-fast-and-slow",
      spineColor: "#fefcf5",
      textColor: "#111111",
      desc: "daniel kahneman is a psychologist who won the nobel prize in economics, which tells you something about how far the ideas in this book reach. the central framework is two systems: system 1 is fast, automatic, intuitive, always running - it's the one that reads emotions on faces, drives a familiar route, fills in 'bread and' with 'butter.' system 2 is slow, deliberate, effortful - it's the one you use when you're doing mental arithmetic or making a careful decision. the problem is system 2 is lazy and system 1 is overconfident, so most of our decisions are made by a fast system that's full of predictable biases. kahneman goes through dozens of them: anchoring, availability bias, the planning fallacy, loss aversion. the one that hit hardest for me was the planning fallacy - we consistently underestimate how long things will take even when we have direct experience of how wrong our estimates usually are. after reading this i started noticing my own system 1 jumping to conclusions constantly. it doesn't really fix it but at least you know when it's happening."
    },
    {
      title: "Dune",
      author: "Frank Herbert",
      coverId: "11481354",
      localCover: "dune",
      spineColor: "#111111",
      textColor: "#f0f0f0",
      desc: "frank herbert wrote dune in 1965 and it's still the best science fiction worldbuilding ever put on paper. the planet arrakis is this desert world that produces the most valuable substance in the universe - spice, which extends life, enables space travel, and basically powers the entire galactic economy. the story follows paul atreides, whose family gets sent to arrakis and then gets absolutely destroyed by court politics, but the real subject of the book is power - how it works, how it corrupts, how people become myths and what that costs them. herbert was thinking about ecology, religion, colonialism, and the danger of charismatic leaders decades before those were fashionable topics. the detail is insane - there's a whole appendix on the ecology of arrakis, the religion, the language. it rewards slow reading. the sequels get progressively stranger and more philosophical, but the first book is just a perfect piece of work. once you've read it you understand why every piece of sci-fi since has been in its shadow."
    },
    {
      title: "The Hitchhiker's Guide to the Galaxy",
      author: "Douglas Adams",
      coverId: "12986869",
      localCover: "hitchhikers-guide",
      spineColor: "#495454",
      textColor: "#f0f0f0",
      desc: "douglas adams wrote this as a radio show in 1978 and somehow it became one of the most beloved sci-fi books ever, which makes sense once you read it because there's nothing else quite like it. the earth gets demolished to make way for a hyperspace bypass on a thursday morning, and arthur dent - an ordinary british guy in his dressing gown - gets accidentally rescued by his friend ford prefect, who turns out to be an alien researcher for the hitchhiker's guide to the galaxy, an electronic encyclopedia whose cover says 'don't panic' in large friendly letters. the whole book is like that. it's comedy, but it's comedy that's actually thinking about big questions - the meaning of life, the nature of consciousness, whether the universe is indifferent or just bureaucratically incompetent. the answer to life the universe and everything is 42, which the characters receive with complete sincerity and then spend the rest of the series trying to figure out what the question was. it's short, it's funny, and it makes you think about existence in a way that's somehow less depressing than most philosophy."
    }
  ];

  function getCover(book) {
    return "/images/reading/" + book.localCover + ".jpg";
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
        coverImg.src = getCover(book);
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
