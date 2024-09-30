document.addEventListener('DOMContentLoaded', function() {
  var backToTopButton = document.getElementById("back-to-top");
  var scrollThreshold = 300;

  window.addEventListener('scroll', function() {
    if (window.pageYOffset > scrollThreshold) {
      backToTopButton.classList.add('show');
    } else {
      backToTopButton.classList.remove('show');
    }
  });

  backToTopButton.addEventListener('click', function(e) {
    e.preventDefault();
    scrollToTop();
  });

  function scrollToTop() {
    var currentPosition = window.pageYOffset;
    var targetPosition = 0;
    var distance = targetPosition - currentPosition;
    var duration = 500;
    var start = null;

    function step(timestamp) {
      if (!start) start = timestamp;
      var progress = timestamp - start;
      var percentage = Math.min(progress / duration, 1);
      window.scrollTo(0, currentPosition + distance * easeInOutCubic(percentage));
      if (progress < duration) {
        window.requestAnimationFrame(step);
      }
    }

    window.requestAnimationFrame(step);
  }

  function easeInOutCubic(t) {
    return t < 0.5 
      ? 4 * t * t * t 
      : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
  }
});