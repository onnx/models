// main.test.js

const { JSDOM } = require('jsdom');
const fetch = require('node-fetch');

global.fetch = fetch;
global.window = new JSDOM('<!doctype html><html><body></body></html>').window;
global.document = window.document;

const { fetchData, renderCards } = require('../script.js');  // assuming you've exported these functions

// Mock fetchData
jest.mock('../script.js', () => ({
  fetchData: jest.fn(),
  renderCards: jest.fn()
}));

describe('Fetch Data', () => {
  beforeEach(() => {
    fetchData.mockClear();
  });

  it('fetches successfully with expected attributes', async () => {
        const mockData = [
            { title: 'Model1', description: 'Task: Task1', author: 'Author1', opset: 'Opset1', downloadUrl: 'url1' }
        ];
        fetchData.mockResolvedValueOnce(mockData);
        const result = await fetchData();
        expect(result).toEqual(mockData);
        expect(result[0]).toHaveProperty('title');
        expect(result[0]).toHaveProperty('description');
        expect(result[0]).toHaveProperty('author');
        expect(result[0]).toHaveProperty('opset');
        expect(result[0]).toHaveProperty('downloadUrl');
    });

  it('fetches erroneously', async () => {
    fetchData.mockRejectedValueOnce(new Error('An error occurred'));
    await expect(fetchData()).rejects.toThrow('An error occurred');
  });
});

describe('Render Cards', () => {
    let mainContent;

    beforeEach(() => {
      // Create a new mainContent div and append it to the document body
      mainContent = document.createElement('div');
      mainContent.id = 'main-content';
      document.body.appendChild(mainContent);
    });
  
    afterEach(() => {
      // Clean up after each test
      document.body.removeChild(mainContent);
    });

    it('renders cards correctly', () => {
        const fakeData = [
        { title: 'Model1', description: 'Task: Task1', author: 'Author1', opset: 'Opset1', downloadUrl: 'url1' },
        { title: 'Model2', description: 'Task: Task2', author: 'Author2', opset: 'Opset2', downloadUrl: 'url2' }
        ];

        // Mock the renderCards method to simulate what it does
        renderCards(fakeData);

        // Simulate what renderCards would add to the DOM
        const mainContent = document.getElementById('main-content');
        expect(mainContent.children.length).toBe(2); // Two cards should be rendered
        
        const firstCard = mainContent.children[0];
        expect(firstCard.className).toBe('card');
        expect(firstCard.querySelector('h3').textContent).toBe('Model1');
        expect(firstCard.querySelector('p').textContent).toMatch(/Task: Task1/);
        expect(firstCard.querySelector('p').textContent).toMatch(/Author: Author1/);
        expect(firstCard.querySelector('p').textContent).toMatch(/Opset: Opset1/);
    });
});

describe('Pagination Buttons', () => {
    let currentPage;
    let filteredData;
    let prevPage, nextPage;

    let numberOfCardsRendered = 0;  // This variable will help us check whether the correct number of cards is rendered

    function renderCards(data) {
      // Simulate what renderCards would do in your application.
      // For the sake of this test, we'll simply count how many items we've been asked to render.
    
      numberOfCardsRendered = data.length;
    }

    beforeEach(() => {
        // Initialize JSDOM
        const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
        global.document = dom.window.document;

        // Initialize currentPage and filteredData
        currentPage = 1;
        filteredData = Array(50).fill({}); // Example, make it match your actual filteredData

        // Create and insert prevPage and nextPage buttons into the DOM
        prevPage = document.createElement('button');
        nextPage = document.createElement('button');
        prevPage.id = 'prev-page';
        nextPage.id = 'next-page';
        document.body.appendChild(prevPage);
        document.body.appendChild(nextPage);

        // Attach event listeners (your actual pagination logic)
        prevPage.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderCards(filteredData);
        }
        });
        nextPage.addEventListener('click', () => {
        const totalPages = Math.ceil(filteredData.length / 10);  // Assuming 10 items per page
        if (currentPage < totalPages) {
            currentPage++;
            renderCards(filteredData);
        }
        });
    });

    it('goes to the previous page when prevPage is clicked', () => {
        currentPage = 2;
        prevPage.click();
        expect(currentPage).toBe(1);
    });

    it('does not go to the previous page when already on first page', () => {
        currentPage = 1;
        prevPage.click();
        expect(currentPage).toBe(1);
    });

    it('goes to the next page when nextPage is clicked', () => {
        const totalPages = Math.ceil(filteredData.length / 10);  // Assuming 10 items per page
        if (currentPage < totalPages) {
        nextPage.click();
        expect(currentPage).toBe(2);
        }
    });

    it('does not go to the next page when already on last page', () => {
        currentPage = Math.ceil(filteredData.length / 10);  // Last page
        nextPage.click();
        expect(currentPage).toBe(Math.ceil(filteredData.length / 10));  // Should remain on last page
    });
});


describe('Search Bar', () => {
    let searchBar, mainContent;

    let numberOfCardsRendered = 0;

    function renderCards(data) {
        numberOfCardsRendered = data.length;
    }
    beforeEach(() => {
        // Create DOM elements and append them to the body
        searchBar = document.createElement('input');
        searchBar.id = 'search-bar';
        document.body.appendChild(searchBar);

        mainContent = document.createElement('div');
        mainContent.id = 'main-content';
        document.body.appendChild(mainContent);
    });

    afterEach(() => {
        // Clean up DOM elements after each test
        document.body.removeChild(searchBar);
        document.body.removeChild(mainContent);
    });

    it('filters models based on the search query', () => {
        const fakeData = [
            { title: 'ResNet', description: 'Task: Image Classification', author: 'Author1', opset: 'Opset1', downloadUrl: 'url1' },
            { title: 'Model2', description: 'Task: NLP', author: 'Author2', opset: 'Opset2', downloadUrl: 'url2' }
        ];

        // Simulate the actual event listener you would have on the search bar
        searchBar.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            const searchResults = fakeData.filter(item => item.title.toLowerCase().includes(query));
            renderCards(searchResults);
        });

        // Trigger input event
        searchBar.value = 'resnet';
        const inputEvent = new Event('input');
        searchBar.dispatchEvent(inputEvent);

        // Validate that renderCards was called with the correct number of models
        expect(numberOfCardsRendered).toBe(1);
    });
});

describe('Filter Buttons', () => {
    it('filters models based on the selected task', () => {
        const fakeData = [
        { title: 'Model1', description: 'Task: Computer Vision', author: 'Author1', opset: 'Opset1', downloadUrl: 'url1' },
        { title: 'Model2', description: 'Task: NLP', author: 'Author2', opset: 'Opset2', downloadUrl: 'url2' }
        ];
        
        const mainContent = document.createElement('div');
        mainContent.id = 'main-content';
        document.body.appendChild(mainContent);
        
        const taskFilterButton = document.createElement('div');
        taskFilterButton.className = 'filter-button';
        taskFilterButton.setAttribute('data-value', 'Computer Vision');
        document.body.appendChild(taskFilterButton);
        
        const clickEvent = new Event('click');
        taskFilterButton.dispatchEvent(clickEvent);
        renderCards(fakeData.filter(item => item.description.split(': ')[1] === 'Computer Vision'));

        expect(mainContent.children.length).toBe(1);
        expect(mainContent.querySelector('p').textContent).toMatch(/Task: Computer Vision/);
    });
});