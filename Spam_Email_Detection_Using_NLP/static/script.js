 const messageInput = document.getElementById("messageInput");
    const resetBtn = document.getElementById("resetBtn");

    function autoResize(textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
    }

    messageInput.addEventListener('input', () => autoResize(messageInput));
    resetBtn.addEventListener('click', () => {
      messageInput.value = "";
      autoResize(messageInput);
    });

    // Initial resize
    autoResize(messageInput);

    // GSAP Animation
    const resultBox = document.getElementById('resultBox');
    const overlay = document.getElementById('overlay');
    const closePopup = document.getElementById('closePopup');

    if (resultBox && overlay) {
      gsap.to(resultBox, { scale: 1, opacity: 1, duration: 2, ease: "power3.out" });

      closePopup.addEventListener('click', () => {
        gsap.to(resultBox, {
          scale: 0.9,
          opacity: 0,
          duration: 0.5,
          ease: "power2.inOut",
          onComplete: () => {
            overlay.style.display = 'none';
          }
        });
      });
    }