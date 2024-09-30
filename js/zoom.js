document.addEventListener('DOMContentLoaded', function() {
    const zoomableImages = document.querySelectorAll('.zoomable');
    const zoomLevel = 2.5;

    zoomableImages.forEach(img => {
        let isZooming = false;

        function startZoom(e) {
            e.preventDefault();
            isZooming = true;
            zoom(e);
            img.classList.add('zoomed');
        }

        function endZoom() {
            isZooming = false;
            resetZoom();
            img.classList.remove('zoomed');
        }

        function zoom(e) {
            if (!isZooming) return;

            const rect = img.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const percentX = x / rect.width;
            const percentY = y / rect.height;

            img.style.transformOrigin = `${percentX * 100}% ${percentY * 100}%`;
            img.style.transform = `scale(${zoomLevel})`;
        }

        function resetZoom() {
            img.style.transformOrigin = 'center center';
            img.style.transform = 'none';
        }

        img.addEventListener('mousedown', startZoom);
        img.addEventListener('mousemove', zoom);
        img.addEventListener('mouseup', endZoom);
        img.addEventListener('mouseleave', endZoom);
        img.addEventListener('dragstart', e => e.preventDefault());
    });

    document.addEventListener('mouseup', function() {
        zoomableImages.forEach(img => {
            img.dispatchEvent(new Event('mouseup'));
        });
    });
});
