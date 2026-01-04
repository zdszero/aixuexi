function saveQuiz(title, url, button) {
  let quizzes = JSON.parse(localStorage.getItem('quizCollection')) || [];
  const isSaved = quizzes.some(quiz => quiz.url === url);

  // Find the quiz content element (adjust selector based on your HTML structure)
  const quizContentElement = button.closest('.quiz-header')?.nextElementSibling;
  console.log(quizContentElement)
  let preview = '';
  if (quizContentElement) {
    // Extract the first 100 characters, stripping HTML tags if markdownified
    preview = quizContentElement.textContent.trim().substring(0, 100);
    if (quizContentElement.textContent.length > 100) preview += '...';
  }

  if (!isSaved) {
    // Save the quiz with preview
    quizzes.push({ title, url, preview });
    localStorage.setItem('quizCollection', JSON.stringify(quizzes));
    button.classList.add('saved');
    button.setAttribute('aria-label', 'Remove Saved Quiz');
    button.querySelector('.quiz-save-icon').innerHTML = '<path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />';
  } else {
    // Unsave the quiz
    quizzes = quizzes.filter(quiz => quiz.url !== url);
    localStorage.setItem('quizCollection', JSON.stringify(quizzes));
    button.classList.remove('saved');
    button.setAttribute('aria-label', 'Save Quiz');
    button.querySelector('.quiz-save-icon').innerHTML = '<path d="M5 4a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v14l-5-2.5L5 18V4Z" />';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.quiz-save-btn').forEach(button => {
    const title = button.getAttribute('data-quiz-id');
    const url = button.getAttribute('data-quiz-url');
    const quizzes = JSON.parse(localStorage.getItem('quizCollection')) || [];

    // Initialize button state based on localStorage
    if (quizzes.some(quiz => quiz.url === url)) {
      button.classList.add('saved');
      button.setAttribute('aria-label', 'Remove Saved Quiz');
      button.querySelector('.quiz-save-icon').innerHTML = '<path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />';
    } else {
      button.classList.remove('saved');
      button.setAttribute('aria-label', 'Save Quiz');
      button.querySelector('.quiz-save-icon').innerHTML = '<path d="M5 4a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v14l-5-2.5L5 18V4Z" />';
    }

    // Attach click event listener
    button.addEventListener('click', () => saveQuiz(title, url, button));
  });
});
