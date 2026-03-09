(function() {
  pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

  var papers = [
    {
      title: "Attention Is All You Need",
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
      shortAuthors: "Vaswani et al., 2017",
      year: "2017",
      tag: "transformers",
      arxiv: "https://arxiv.org/abs/1706.03762",
      pdf: "/papers/1706.03762.pdf",
      desc: "this is literally the paper that started everything. vaswani and the team at google proposed the transformer architecture - a model based entirely on attention mechanisms, ditching recurrence and convolutions completely. the core idea is self-attention, where every token in a sequence can directly attend to every other token, making it way easier to capture long-range dependencies. they introduced multi-head attention so the model can jointly attend to information from different representation subspaces. the results on machine translation were insane for the time, and honestly the architecture ended up being the foundation for basically every major language model since - BERT, GPT, all of it. the paper is dense but super well-written, and re-reading it now knowing what came after makes it even more impressive how much they got right from the start."
    },
    {
      title: "StepCoder: Improve Code Generation with Reinforcement Learning from Compiler Feedback",
      authors: "Shihan Dou, Yan Liu, Haoxiang Jia, Limao Xiong, Enyu Zhou, Wei Shen, Junjie Shan, Caishuang Huang, Xiao Wang, Xiaoran Fan, Zhiheng Xi, Yuhao Zhou, Tao Ji, Rui Zheng, Qi Zhang, Xuanjing Huang, Tao Gui",
      shortAuthors: "Dou et al., 2024",
      year: "2024",
      tag: "code gen",
      arxiv: "https://arxiv.org/abs/2402.01391",
      pdf: "/papers/2402.01391.pdf",
      desc: "this paper tackles a really practical problem - how do you make LLMs generate better code using actual compiler feedback as a reward signal. the key insight is that long code generation makes RL exploration super hard, so they break it down into a curriculum of code completion subtasks (CCCS) where the model gradually learns to write longer and more complex code. they also have this fine-grained optimization thing (FGO) that masks out code that never actually gets executed during training, so the model focuses on the parts that matter. it's a clever combination of curriculum learning and RL that makes the whole training process way more tractable. the results show solid improvements on code generation benchmarks, and the approach feels like a natural evolution of how we should be training code models - using the compiler as the ultimate judge of correctness."
    },
    {
      title: "UIClip: A Data-driven Model for Assessing User Interface Design",
      authors: "Jason Wu, Yi-Hao Peng, Xin Yue Amanda Li, Amanda Swearngin, Jeffrey P. Bigham, Jeffrey Nichols",
      shortAuthors: "Wu et al., 2024",
      year: "2024",
      tag: "ui / hci",
      arxiv: "https://arxiv.org/abs/2404.12500",
      pdf: "/papers/UIClip_A_Data_driven_Model_for_Assessing_User_Interface_Design_2404.pdf",
      desc: "this one is super interesting because it tries to solve something that's traditionally been really subjective - evaluating whether a UI design is actually good. the team built UIClip, which is basically a CLIP-style model fine-tuned to assess design quality and visual relevance of user interfaces given a screenshot and a text description. what makes it work is the training data - they combined automated crawling, synthetic augmentation, and actual human ratings to build a massive dataset of UIs ranked by quality. when they compared UIClip's judgments against ratings from 12 human designers, it had the highest agreement with ground truth compared to other baselines. it's a really cool intersection of ML and HCI, and honestly feels like it could be super useful for automated design review or helping people who aren't designers build better-looking interfaces."
    },
    {
      title: "Eight Things to Know about Large Language Models",
      authors: "Samuel R. Bowman",
      shortAuthors: "Bowman, 2023",
      year: "2023",
      tag: "llms",
      arxiv: "https://arxiv.org/abs/2304.00612",
      pdf: "/papers/2304.00612.pdf",
      desc: "samuel bowman from nyu wrote this really solid overview of what we actually know about LLMs versus all the hype and speculation. he lays out eight key points - like how LLMs predictably get better with more compute and data, but many important behaviors emerge unpredictably at certain scales. he argues they actually do seem to learn representations of the real world, not just surface statistics, which is a big deal for the whole 'stochastic parrots' debate. he also gets into how we still have no reliable way to steer their behavior or prevent them from saying stuff we don't want. it's one of those papers that cuts through the noise and just presents what the evidence actually shows, which is refreshing when everyone else is either panicking or overhyping these things."
    },
    {
      title: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
      authors: "Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning",
      shortAuthors: "Sarthi et al., 2024",
      year: "2024",
      tag: "rag",
      arxiv: "https://arxiv.org/abs/2401.18059",
      pdf: "/papers/2401.18059.pdf",
      desc: "this paper introduces a really clever approach to retrieval-augmented generation. instead of just chunking documents and doing basic similarity search like most RAG systems, RAPTOR recursively embeds, clusters, and summarizes text chunks to build a tree structure with different levels of abstraction. so at the leaves you have the original chunks, but as you go up the tree you get increasingly high-level summaries. at inference time it can pull from any level of the tree, which means it can grab both specific details and broad context depending on what the question needs. they showed a 20% improvement on the QuALITY benchmark when paired with GPT-4, which is pretty wild. honestly it's one of those ideas that feels obvious in hindsight - of course you should organize your retrieval hierarchically instead of treating everything as flat chunks."
    },
    {
      title: "UICoder: Finetuning Large Language Models to Generate User Interface Code through Automated Feedback",
      authors: "Jason Wu, Eldon Schoop, Alan Leung, Titus Barik, Jeffrey P. Bigham, Jeffrey Nichols",
      shortAuthors: "Wu et al., 2024",
      year: "2024",
      tag: "ui / code gen",
      arxiv: "https://arxiv.org/abs/2406.07739",
      pdf: "/papers/2406.07739v1.pdf",
      desc: "jason wu and team tackle the problem of getting LLMs to generate UI code that actually compiles and looks decent. the usual approach is either expensive human feedback or distilling from proprietary models like GPT-4, but they go a different route - using automated feedback from compilers and multimodal models to iteratively improve an open-source LLM. they self-generate a huge synthetic dataset, then aggressively filter and score the outputs using automated tools to keep only the good stuff. it's basically a self-improvement loop where the model keeps getting better at generating working UI code without needing humans in the loop. super practical approach and honestly feels like the right direction for domain-specific code generation - let the tools that evaluate the output be the teachers."
    },
    {
      title: "Flowy: Supporting UX Design Decisions Through AI-Driven Pattern Annotation in Multi-Screen User Flows",
      authors: "Yuwen Lu, Ziang Tong, Qinyi Zhao, Yewon Oh, Bryan Wang, Toby Jia-Jun Li",
      shortAuthors: "Lu et al., 2024",
      year: "2024",
      tag: "ux / ai",
      arxiv: "https://arxiv.org/abs/2406.16177",
      pdf: "/papers/2406.16177v1.pdf",
      desc: "most AI design tools focus on generating single static screens, but flowy goes after the much harder problem of multi-screen user flows - like how do you design the entire experience of signing up, onboarding, and navigating an app. they built this tool that uses large multimodal models to automatically identify and annotate UX design patterns across sequences of screens, helping designers understand and reuse proven interaction patterns. it's built on a high-quality user flow dataset and augments the designer's ideation process rather than trying to replace it. what i find cool is that it treats design as a flow problem rather than a screen-by-screen thing, which is way closer to how users actually experience products. the AI helps you see patterns you might miss when you're too deep in the details of individual screens."
    },
    {
      title: "Beyond Labels: Leveraging Deep Learning and LLMs for Content Metadata",
      authors: "Saurabh Agrawal, John Trenkle, Jaya Kawale",
      shortAuthors: "Agrawal et al., 2023",
      year: "2023",
      tag: "recsys",
      arxiv: "https://arxiv.org/abs/2309.08787",
      pdf: "/papers/2309.08787.pdf",
      desc: "this paper digs into a really practical problem in recommender systems - how content metadata like genre labels can be noisy, incomplete, and inconsistent, and how that messes up recommendations. instead of relying on those messy human-assigned labels, they use deep learning and LLMs to generate richer, more consistent metadata from the content itself. it's the kind of work that doesn't get as much hype as the flashy generative AI stuff but is super important for anyone actually building recommendation systems at scale. the approach is pragmatic - they're not trying to reinvent the wheel, just showing that you can get way better metadata by letting models analyze the actual content rather than trusting whatever labels someone slapped on it. feels very relevant for streaming platforms and content discovery in general."
    }
  ];

  var paperShelf = document.getElementById("paper-shelf");
  var overlay = document.getElementById("paper-modal-overlay");
  var thumbCache = {};

  function renderThumb(canvas, pdfUrl) {
    if (thumbCache[pdfUrl]) {
      canvas.width = thumbCache[pdfUrl].width;
      canvas.height = thumbCache[pdfUrl].height;
      canvas.getContext("2d").drawImage(thumbCache[pdfUrl], 0, 0);
      return;
    }
    pdfjsLib.getDocument(pdfUrl).promise.then(function(pdf) {
      pdf.getPage(1).then(function(page) {
        var containerWidth = canvas.parentElement.offsetWidth;
        var unscaledViewport = page.getViewport({ scale: 1 });
        var scale = (containerWidth * 2) / unscaledViewport.width;
        var viewport = page.getViewport({ scale: scale });
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        page.render({ canvasContext: canvas.getContext("2d"), viewport: viewport }).promise.then(function() {
          thumbCache[pdfUrl] = canvas;
        });
      });
    });
  }

  function buildShelf() {
    paperShelf.innerHTML = "";
    for (var i = 0; i < papers.length; i++) {
      (function(index) {
        var paper = papers[index];

        var card = document.createElement("div");
        card.className = "paper-card";

        var thumbWrap = document.createElement("div");
        thumbWrap.className = "paper-thumb";
        var thumbCanvas = document.createElement("canvas");
        thumbWrap.appendChild(thumbCanvas);

        var info = document.createElement("div");
        info.className = "paper-card-info";

        var title = document.createElement("p");
        title.className = "paper-title";
        title.textContent = paper.title;

        var authors = document.createElement("p");
        authors.className = "paper-authors";
        authors.textContent = paper.shortAuthors;

        info.appendChild(title);
        info.appendChild(authors);
        card.appendChild(thumbWrap);
        card.appendChild(info);

        card.onclick = function() { openModal(paper); };

        paperShelf.appendChild(card);
        renderThumb(thumbCanvas, paper.pdf);
      })(i);
    }
  }

  function openModal(paper) {
    overlay.innerHTML = '';
    var modal = document.createElement("div");
    modal.className = "paper-modal";

    var header = document.createElement("div");
    header.className = "paper-modal-header";

    var headerInfo = document.createElement("div");
    headerInfo.className = "paper-modal-header-info";
    headerInfo.innerHTML =
      '<h2>' + paper.title + '</h2>' +
      '<p class="modal-authors">' + paper.authors + ' \u00b7 ' + paper.year + '</p>' +
      '<div class="modal-links">' +
        '<a href="' + paper.arxiv + '" target="_blank" rel="noopener">arXiv</a>' +
        '<a href="' + paper.pdf + '" target="_blank" rel="noopener">Download PDF</a>' +
      '</div>';

    var closeBtn = document.createElement("button");
    closeBtn.className = "paper-modal-close";
    closeBtn.innerHTML = "\u2715";
    closeBtn.onclick = closeModal;

    header.appendChild(headerInfo);
    header.appendChild(closeBtn);

    var body = document.createElement("div");
    body.className = "paper-modal-body";

    var pages = document.createElement("div");
    pages.className = "paper-modal-pages";
    pages.innerHTML = '<p class="loading-pages">Loading paper...</p>';
    body.appendChild(pages);

    modal.appendChild(header);
    modal.appendChild(body);
    overlay.appendChild(modal);
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";

    pdfjsLib.getDocument(paper.pdf).promise.then(function(pdf) {
      pages.innerHTML = "";
      var totalPages = pdf.numPages;
      for (var p = 1; p <= totalPages; p++) {
        (function(pageNum) {
          pdf.getPage(pageNum).then(function(page) {
            var scale = 2;
            var viewport = page.getViewport({ scale: scale });
            var canvas = document.createElement("canvas");
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            canvas.style.order = pageNum;
            page.render({ canvasContext: canvas.getContext("2d"), viewport: viewport });

            var inserted = false;
            var children = pages.children;
            for (var c = 0; c < children.length; c++) {
              if (parseInt(children[c].style.order) > pageNum) {
                pages.insertBefore(canvas, children[c]);
                inserted = true;
                break;
              }
            }
            if (!inserted) pages.appendChild(canvas);
          });
        })(p);
      }
    });
  }

  function closeModal() {
    overlay.classList.remove("open");
    overlay.innerHTML = "";
    document.body.style.overflow = "";
  }

  overlay.addEventListener("click", function(e) {
    if (e.target === overlay) closeModal();
  });

  document.addEventListener("keydown", function(e) {
    if (e.key === "Escape") closeModal();
  });

  buildShelf();
})();
