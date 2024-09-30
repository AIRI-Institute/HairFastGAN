document.addEventListener('DOMContentLoaded', (event) => {
    const copyButton = document.getElementById('copy-button');
    const bibtexContent = document.getElementById('bibtex-content');

    copyButton.addEventListener('click', () => {
        navigator.clipboard.writeText(bibtexContent.textContent.trim()).then(() => {
            const originalText = copyButton.innerHTML;
            copyButton.innerHTML = '<svg class="copy-icon" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path></svg><span>Copied!</span>';
            setTimeout(() => {
                copyButton.innerHTML = originalText;
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    });
});
