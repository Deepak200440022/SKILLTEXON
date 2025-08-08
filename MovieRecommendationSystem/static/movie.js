document.addEventListener("DOMContentLoaded", function () {
  const btn = document.getElementById("toggle-cast-btn");
  const castSection = document.getElementById("cast-section");
  let expanded = false;

  function toggleCast() {
    const extras = document.querySelectorAll(".extra-cast");
    expanded = !expanded;
    extras.forEach(row => row.style.display = expanded ? "table-row" : "none");
    btn.innerText = expanded ? "Show Less" : "Show Full Cast";
    if (expanded) {
      window.addEventListener("scroll", checkPosition);
      checkPosition();
    } else {
      window.removeEventListener("scroll", checkPosition);
      btn.classList.remove("floating");
    }
  }

  function checkPosition() {
    const castBottom = castSection.getBoundingClientRect().bottom;
    if (castBottom < window.innerHeight) {
      btn.classList.remove("floating");
    } else {
      btn.classList.add("floating");
    }
  }

  btn.addEventListener("click", toggleCast);
});