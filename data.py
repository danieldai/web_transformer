# Test dataset with ground truth labels
TEST_DATASET = [
    # HR Questions
    {"question": "How many leaves do I have?", "expected": "HR"},
    {"question": "How many vacation days do I have left?", "expected": "HR"},
    {"question": "What's the maternity leave policy?", "expected": "HR"},
    {"question": "When will I receive my salary this month?", "expected": "HR"},
    {"question": "How do I update my benefits enrollment?", "expected": "HR"},
    {"question": "Who do I contact about workplace harassment?", "expected": "HR"},
    {"question": "What's the process for requesting a promotion?", "expected": "HR"},
    {"question": "How do I check my performance review?", "expected": "HR"},

    # IT Questions
    {"question": "I can't access the VPN", "expected": "IT"},
    {"question": "My laptop won't start", "expected": "IT"},
    {"question": "I forgot my email password", "expected": "IT"},
    {"question": "The printer on 3rd floor isn't working", "expected": "IT"},
    {"question": "Can I get access to the shared drive?", "expected": "IT"},
    {"question": "My computer is running very slow", "expected": "IT"},
    {"question": "I need software installed on my machine", "expected": "IT"},
    {"question": "WiFi connection keeps dropping", "expected": "IT"},

    # Sales Questions
    {"question": "What's my Q4 sales quota?", "expected": "Sales"},
    {"question": "How do I update a lead in the CRM?", "expected": "Sales"},
    {"question": "What's the status of the ABC Corp deal?", "expected": "Sales"},
    {"question": "Can you send me the latest sales pipeline report?", "expected": "Sales"},
    {"question": "How do I qualify a new prospect?", "expected": "Sales"},
    {"question": "What's our pricing for enterprise customers?", "expected": "Sales"},
    {"question": "I need the sales deck for tomorrow's pitch", "expected": "Sales"},

    # Finance Questions
    {"question": "How do I submit an expense report?", "expected": "Finance"},
    {"question": "When will my reimbursement be processed?", "expected": "Finance"},
    {"question": "What's the budget for my department this quarter?", "expected": "Finance"},
    {"question": "I need approval for a $5000 purchase", "expected": "Finance"},
    {"question": "Can you send me the latest financial statements?", "expected": "Finance"},
    {"question": "How do I process an invoice from a vendor?", "expected": "Finance"},
    {"question": "What's the company's revenue this year?", "expected": "Finance"},

    # Operations Questions
    {"question": "What's the status of shipment #12345?", "expected": "Operations"},
    {"question": "We're running low on inventory for product X", "expected": "Operations"},
    {"question": "How do I schedule a delivery to a customer?", "expected": "Operations"},
    {"question": "Can we expedite this order?", "expected": "Operations"},
    {"question": "I need to update our supplier contact information", "expected": "Operations"},
    {"question": "What's the lead time for manufacturing?", "expected": "Operations"},
    {"question": "Meeting room booking system isn't working", "expected": "Operations"},

    # Legal Questions
    {"question": "I need a contract reviewed", "expected": "Legal"},
    {"question": "Can you send me the standard NDA template?", "expected": "Legal"},
    {"question": "What are our GDPR compliance requirements?", "expected": "Legal"},
    {"question": "Is this clause in the agreement enforceable?", "expected": "Legal"},
    {"question": "How do we handle intellectual property disputes?", "expected": "Legal"},
    {"question": "Do we need legal approval for this partnership?", "expected": "Legal"},

    # Marketing Questions
    {"question": "What's the performance of our latest campaign?", "expected": "Marketing"},
    {"question": "Can you send me the brand guidelines?", "expected": "Marketing"},
    {"question": "When is the next product launch?", "expected": "Marketing"},
    {"question": "I need content for our social media posts", "expected": "Marketing"},
    {"question": "What's our target audience for this campaign?", "expected": "Marketing"},
    {"question": "Can we get market research data for segment X?", "expected": "Marketing"},

    # Customer Support Questions
    {"question": "Customer is complaining about a defective product", "expected": "Customer Support"},
    {"question": "How do I process a refund for order #789?", "expected": "Customer Support"},
    {"question": "Customer wants to return an item", "expected": "Customer Support"},
    {"question": "User can't log into their account", "expected": "Customer Support"},
    {"question": "Customer is asking about shipping times", "expected": "Customer Support"},
    {"question": "Product isn't working as described", "expected": "Customer Support"},
    {"question": "Customer wants to cancel their subscription", "expected": "Customer Support"},
]