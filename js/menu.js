document.addEventListener('DOMContentLoaded', function() {
    const menu = document.getElementById('quick-access-menu');
    if (!menu) return;
  
    const menuLinks = menu.getElementsByTagName('a');
    const sections = Array.from(document.querySelectorAll('section[id], div[id="bibtex"]'));
  
    function isPageBottom() {
      return (window.innerHeight + window.scrollY) >= document.body.offsetHeight - 50;
    }
  
    function getCurrentSection() {
      if (isPageBottom()) {
        return sections[sections.length - 1].id;
      }
  
      const scrollPosition = window.scrollY;
  
      for (let i = sections.length - 1; i >= 0; i--) {
        const section = sections[i];
        const sectionTop = section.offsetTop - 100; // Добавляем отступ для более раннего переключения
  
        if (scrollPosition >= sectionTop) {
          return section.id;
        }
      }
  
      return sections[0].id;
    }
  
    function highlightMenuLink() {
      const currentSection = getCurrentSection();
  
      Array.from(menuLinks).forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').slice(1) === currentSection) {
          link.classList.add('active');
        }
      });
    }
  
    // Highlight menu link on scroll
    window.addEventListener('scroll', highlightMenuLink);
  
    // Initial highlight
    highlightMenuLink();
  
    // Smooth scroll to section when clicking menu items
    Array.from(menuLinks).forEach(link => {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').slice(1);
        const targetSection = document.getElementById(targetId);
        
        if (targetSection) {
          window.scrollTo({
            top: targetSection.offsetTop - 50, // Добавляем отступ для лучшего позиционирования
            behavior: 'smooth'
          });
        } else {
          console.warn(`Section with id "${targetId}" not found`);
        }
      });
    });
  });