 const container = document.getElementById('searchContainer');
    const btn = document.getElementById('searchBtn');
    const input = document.getElementById('searchInput');
    const suggestionsBox = document.getElementById('suggestionsBox');

    let currentIndex = -1;
    let currentSuggestions = [];

    btn.addEventListener('click', () => {
      container.classList.add('expanded');
      btn.style.opacity = '0';
      setTimeout(() => {
        btn.style.display = 'none';
        input.style.display = 'block';
        input.focus();
      }, 300);
    });

    input.addEventListener('input', async function () {
      const query = this.value.toLowerCase().trim();
      suggestionsBox.innerHTML = "";
      currentIndex = -1;
      currentSuggestions = [];
      if (!query) return;

      try {
        const res = await fetch(`/suggest?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        currentSuggestions = data;

        data.forEach((movie, i) => {
          const item = document.createElement("div");
          item.className = "suggestion-item";
          item.setAttribute("data-id", movie.id);
          item.setAttribute("data-index", i);
          item.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
              <img src="https://image.tmdb.org/t/p/original/${movie.poster_path}"  alt="${movie.title}" style="width: 40px; height: 60px; object-fit: cover; border-radius: 4px;">
              <span>${movie.title}</span>
            </div>
          `;
          item.addEventListener("click", () => {
            window.location.href = `/movie/${movie.id}`;
          });
          suggestionsBox.appendChild(item);
        });
      } catch (error) {
        console.error("Suggestion fetch failed:", error);
      }
    });

    input.addEventListener("keydown", (e) => {
      const items = suggestionsBox.querySelectorAll(".suggestion-item");
      if (!items.length) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        currentIndex = (currentIndex + 1) % items.length;
        updateHighlight(items);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        currentIndex = (currentIndex - 1 + items.length) % items.length;
        updateHighlight(items);
      } else if (e.key === "Enter" && currentIndex >= 0) {
        e.preventDefault();
        const selected = currentSuggestions[currentIndex];
        if (selected && selected.id) {
          window.location.href = `/movie/${selected.id}`;
        }
      }
    });

    function updateHighlight(items) {
      items.forEach(item => item.classList.remove("highlight"));
      if (currentIndex >= 0 && items[currentIndex]) {
        items[currentIndex].classList.add("highlight");
        items[currentIndex].scrollIntoView({ block: "nearest" });
      }
    }

    document.addEventListener('click', (e) => {
      if (!container.contains(e.target)) {
        if (input.style.display === 'block') {
          suggestionsBox.innerHTML = "";
          input.style.display = 'none';
          btn.style.display = 'block';
          btn.style.opacity = '1';
          container.classList.remove('expanded');
        }
      }
    });