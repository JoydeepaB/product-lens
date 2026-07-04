import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ============ EMBEDDED DATA - NO EXTERNAL FILES ============

PRODUCTS_DATA = [
    {"id": 1, "name": "Nubex Wireless Bluetooth Earbuds", "category": "Electronics", "description": "Premium Wireless Bluetooth Earbuds by Nubex. Features HD display, sweat resistant design, touch controls and fast charging support. Ideal for everyday use with reliable performance.", "price": 499},
    {"id": 2, "name": "Nexura Smartphone 128GB", "category": "Electronics", "description": "Advanced Smartphone 128GB by Nexura. Features water resistant, USB-C port, long battery life and lightweight build. Ideal for everyday use with reliable performance.", "price": 499},
    {"id": 3, "name": "Aeroline Laptop 15.6 inch", "category": "Electronics", "description": "Lightweight Laptop 15.6 inch by Aeroline. Features touch controls, long battery life, sweat resistant design and noise isolation. Ideal for everyday use with reliable performance.", "price": 5999},
    {"id": 4, "name": "Blustar Smartwatch Fitness Tracker", "category": "Electronics", "description": "Lightweight Smartwatch Fitness Tracker by Blustar. Features lightweight build, HD display, long battery life and USB-C port. Ideal for everyday use with reliable performance.", "price": 799},
    {"id": 5, "name": "Blustar Portable Power Bank 20000mAh", "category": "Electronics", "description": "Ergonomic Portable Power Bank 20000mAh by Blustar. Features HD display, Bluetooth 5.3 connectivity, sweat resistant design and USB-C port. Ideal for everyday use with reliable performance.", "price": 1999},
    {"id": 6, "name": "Nubex Noise Cancelling Headphones", "category": "Electronics", "description": "Advanced Noise Cancelling Headphones by Nubex. Features USB-C port, fast charging support, noise isolation and water resistant. Ideal for everyday use with reliable performance.", "price": 1999},
    {"id": 7, "name": "Solace 4K Action Camera", "category": "Electronics", "description": "Durable 4K Action Camera by Solace. Features long battery life, lightweight build, fast charging support and sweat resistant design. Ideal for everyday use with reliable performance.", "price": 499},
    {"id": 8, "name": "Nexura Gaming Mouse RGB", "category": "Electronics", "description": "Durable Gaming Mouse RGB by Nexura. Features water resistant, noise isolation, sweat resistant design and touch controls. Ideal for everyday use with reliable performance.", "price": 499},
    {"id": 9, "name": "Zentro Mechanical Keyboard", "category": "Electronics", "description": "Lightweight Mechanical Keyboard by Zentro. Features HD display, fast charging support, sweat resistant design and USB-C port. Ideal for everyday use with reliable performance.", "price": 499},
    {"id": 10, "name": "Blustar Tablet 10 inch", "category": "Electronics", "description": "Durable Tablet 10 inch by Blustar. Features lightweight build, noise isolation, Bluetooth 5.3 connectivity and water resistant. Ideal for everyday use with reliable performance.", "price": 1999},
    {"id": 11, "name": "Aeroline Cotton Casual T-Shirt", "category": "Fashion", "description": "Durable Cotton Casual T-Shirt by Aeroline. Features machine washable, slim fit design, all-season wear and anti-wrinkle fabric. Ideal for everyday use with reliable performance.", "price": 3999},
    {"id": 12, "name": "Blustar Slim Fit Denim Jeans", "category": "Fashion", "description": "Durable Slim Fit Denim Jeans by Blustar. Features adjustable straps, all-season wear, comfortable fit and trendy design. Ideal for everyday use with reliable performance.", "price": 299},
    {"id": 13, "name": "Aeroline Running Shoes", "category": "Fashion", "description": "Premium Running Shoes by Aeroline. Features comfortable fit, trendy design, durable stitching and breathable fabric. Ideal for everyday use with reliable performance.", "price": 999},
    {"id": 14, "name": "Solace Leather Wallet", "category": "Fashion", "description": "Ergonomic Leather Wallet by Solace. Features all-season wear, sweat-wicking material, trendy design and comfortable fit. Ideal for everyday use with reliable performance.", "price": 3999},
    {"id": 15, "name": "Corevo Formal Shirt", "category": "Fashion", "description": "Durable Formal Shirt by Corevo. Features slim fit design, all-season wear, durable stitching and comfortable fit. Ideal for everyday use with reliable performance.", "price": 9999},
    {"id": 16, "name": "Blustar Winter Jacket", "category": "Fashion", "description": "Professional Winter Jacket by Blustar. Features trendy design, comfortable fit, all-season wear and machine washable. Ideal for everyday use with reliable performance.", "price": 5999},
    {"id": 17, "name": "Optiva Sunglasses UV Protection", "category": "Fashion", "description": "Advanced Sunglasses UV Protection by Optiva. Features breathable fabric, machine washable, slim fit design and comfortable fit. Ideal for everyday use with reliable performance.", "price": 799},
    {"id": 18, "name": "Blustar Analog Wrist Watch", "category": "Fashion", "description": "Professional Analog Wrist Watch by Blustar. Features machine washable, trendy design, adjustable straps and durable stitching. Ideal for everyday use with reliable performance.", "price": 3999},
    {"id": 19, "name": "Nexura Backpack Laptop Bag", "category": "Fashion", "description": "Durable Backpack Laptop Bag by Nexura. Features adjustable straps, breathable fabric, machine washable and comfortable fit. Ideal for everyday use with reliable performance.", "price": 5999},
    {"id": 20, "name": "Vantix Sports Sneakers", "category": "Fashion", "description": "Ergonomic Sports Sneakers by Vantix. Features machine washable, durable stitching, trendy design and anti-wrinkle fabric. Ideal for everyday use with reliable performance.", "price": 3999},
    {"id": 21, "name": "Zentro Non-stick Cookware Set", "category": "Home", "description": "Durable Non-stick Cookware Set by Zentro. Features eco-friendly material, compact design, energy efficient and rust resistant. Ideal for everyday use with reliable performance.", "price": 1499},
    {"id": 22, "name": "Nexura LED Desk Lamp", "category": "Home", "description": "Professional LED Desk Lamp by Nexura. Features durable build quality, compact design, space saving and rust resistant. Ideal for everyday use with reliable performance.", "price": 799},
    {"id": 23, "name": "Nexura Memory Foam Pillow", "category": "Home", "description": "Portable Memory Foam Pillow by Nexura. Features easy to clean, space saving, adjustable settings and modern finish. Ideal for everyday use with reliable performance.", "price": 499},
    {"id": 24, "name": "Primeo Air Purifier", "category": "Home", "description": "Durable Air Purifier by Primeo. Features durable build quality, easy to clean, modern finish and quiet operation. Ideal for everyday use with reliable performance.", "price": 499},
    {"id": 25, "name": "Nubex Electric Kettle", "category": "Home", "description": "Eco-Friendly Electric Kettle by Nubex. Features energy efficient, eco-friendly material, compact design and modern finish. Ideal for everyday use with reliable performance.", "price": 3999},
    {"id": 26, "name": "Nexura Vacuum Cleaner", "category": "Home", "description": "Compact Vacuum Cleaner by Nexura. Features quiet operation, eco-friendly material, rust resistant and energy efficient. Ideal for everyday use with reliable performance.", "price": 5999},
    {"id": 27, "name": "Aeroline Study Table Wooden", "category": "Home", "description": "Durable Study Table Wooden by Aeroline. Features rust resistant, space saving, adjustable settings and quiet operation. Ideal for everyday use with reliable performance.", "price": 3999},
    {"id": 28, "name": "Nubex Bedsheet Cotton King Size", "category": "Home", "description": "Lightweight Bedsheet Cotton King Size by Nubex. Features durable build quality, energy efficient, space saving and easy to clean. Ideal for everyday use with reliable performance.", "price": 9999},
    {"id": 29, "name": "Nexura Wall Clock Modern", "category": "Home", "description": "Lightweight Wall Clock Modern by Nexura. Features modern finish, durable build quality, easy to clean and adjustable settings. Ideal for everyday use with reliable performance.", "price": 299},
    {"id": 30, "name": "Aeroline Storage Organizer Box", "category": "Home", "description": "Advanced Storage Organizer Box by Aeroline. Features easy to clean, space saving, energy efficient and quiet operation. Ideal for everyday use with reliable performance.", "price": 999},
    {"id": 31, "name": "Vantix Face Wash Neem Extract", "category": "Beauty", "description": "Eco-Friendly Face Wash Neem Extract by Vantix. Features long lasting formula, travel friendly, suitable for all skin types and cruelty free. Ideal for everyday use with reliable performance.", "price": 9999},
    {"id": 32, "name": "Solace Vitamin C Serum", "category": "Beauty", "description": "Eco-Friendly Vitamin C Serum by Solace. Features long lasting formula, quick absorption, natural ingredients and paraben free. Ideal for everyday use with reliable performance.", "price": 499},
    {"id": 33, "name": "Nubex Sunscreen SPF 50", "category": "Beauty", "description": "High-Performance Sunscreen SPF 50 by Nubex. Features cruelty free, natural ingredients, travel friendly and long lasting formula. Ideal for everyday use with reliable performance.", "price": 299},
    {"id": 34, "name": "Nubex Hair Dryer", "category": "Beauty", "description": "Premium Hair Dryer by Nubex. Features natural ingredients, cruelty free, paraben free and quick absorption. Ideal for everyday use with reliable performance.", "price": 999},
    {"id": 35, "name": "Aeroline Lipstick Matte Finish", "category": "Beauty", "description": "Portable Lipstick Matte Finish by Aeroline. Features quick absorption, suitable for all skin types, natural ingredients and paraben free. Ideal for everyday use with reliable performance.", "price": 1499},
    {"id": 36, "name": "Optiva Moisturizer Cream", "category": "Beauty", "description": "Lightweight Moisturizer Cream by Optiva. Features paraben free, quick absorption, fragrance free option and dermatologically tested. Ideal for everyday use with reliable performance.", "price": 5999},
    {"id": 37, "name": "Zentro Shampoo Anti Dandruff", "category": "Beauty", "description": "Advanced Shampoo Anti Dandruff by Zentro. Features long lasting formula, suitable for all skin types, natural ingredients and fragrance free option. Ideal for everyday use with reliable performance.", "price": 3999},
    {"id": 38, "name": "Aeroline Nail Polish Set", "category": "Beauty", "description": "High-Performance Nail Polish Set by Aeroline. Features dermatologically tested, suitable for all skin types, natural ingredients and fragrance free option. Ideal for everyday use with reliable performance.", "price": 2499},
    {"id": 39, "name": "Vantix Perfume Eau de Parfum", "category": "Beauty", "description": "Eco-Friendly Perfume Eau de Parfum by Vantix. Features non-greasy texture, natural ingredients, quick absorption and paraben free. Ideal for everyday use with reliable performance.", "price": 999},
    {"id": 40, "name": "Vantix Face Mask Sheet Pack", "category": "Beauty", "description": "Lightweight Face Mask Sheet Pack by Vantix. Features dermatologically tested, travel friendly, fragrance free option and cruelty free. Ideal for everyday use with reliable performance.", "price": 1999},
]

REVIEWS_DATA = [
    {"id": 1, "product_id": 1, "reviewer": "Priya K.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-06-11"},
    {"id": 2, "product_id": 1, "reviewer": "Priya K.", "rating": 5, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-06-07"},
    {"id": 3, "product_id": 1, "reviewer": "Amit V.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-05-03"},
    {"id": 4, "product_id": 1, "reviewer": "Kavya H.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-06"},
    {"id": 5, "product_id": 1, "reviewer": "Priya K.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-06-10"},
    {"id": 6, "product_id": 2, "reviewer": "Vikram T.", "rating": 4, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-06-05"},
    {"id": 7, "product_id": 2, "reviewer": "Neha J.", "rating": 4, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-06-06"},
    {"id": 8, "product_id": 2, "reviewer": "Karan D.", "rating": 4, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-06-07"},
    {"id": 9, "product_id": 2, "reviewer": "Karan D.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-06-09"},
    {"id": 10, "product_id": 2, "reviewer": "Manish C.", "rating": 5, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-05-30"},
    {"id": 11, "product_id": 2, "reviewer": "Divya P.", "rating": 4, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-06-14"},
    {"id": 12, "product_id": 2, "reviewer": "Arjun B.", "rating": 5, "text": "Good product nice", "date": "2026-05-20"},
    {"id": 13, "product_id": 2, "reviewer": "Manish C.", "rating": 5, "text": "Nice", "date": "2026-05-20"},
    {"id": 14, "product_id": 2, "reviewer": "Amit V.", "rating": 5, "text": "Nice product", "date": "2026-05-20"},
    {"id": 15, "product_id": 2, "reviewer": "Kavya H.", "rating": 5, "text": "Good", "date": "2026-05-20"},
    {"id": 16, "product_id": 2, "reviewer": "Arjun B.", "rating": 5, "text": "Good", "date": "2026-05-20"},
    {"id": 17, "product_id": 3, "reviewer": "Pooja L.", "rating": 5, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-22"},
    {"id": 18, "product_id": 3, "reviewer": "Amit V.", "rating": 4, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-06-23"},
    {"id": 19, "product_id": 3, "reviewer": "Amit V.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-06-14"},
    {"id": 20, "product_id": 3, "reviewer": "Sanjay N.", "rating": 5, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-05-19"},
    {"id": 21, "product_id": 3, "reviewer": "Manish C.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-05-02"},
    {"id": 22, "product_id": 3, "reviewer": "Karan D.", "rating": 5, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-04"},
    {"id": 23, "product_id": 4, "reviewer": "Ritu W.", "rating": 2, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-06-25"},
    {"id": 24, "product_id": 4, "reviewer": "Rohan G.", "rating": 5, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-06-05"},
    {"id": 25, "product_id": 4, "reviewer": "Rohan G.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-06-25"},
    {"id": 26, "product_id": 4, "reviewer": "Vikram T.", "rating": 2, "text": "Size/fit was off compared to the description, had to return it.", "date": "2026-05-15"},
    {"id": 27, "product_id": 4, "reviewer": "Rahul S.", "rating": 5, "text": "Awesome product awesome price buy buy buy limited stock hurry", "date": "2026-05-10"},
    {"id": 28, "product_id": 4, "reviewer": "Anjali M.", "rating": 5, "text": "Superb quality superb price superb delivery best seller must buy", "date": "2026-05-10"},
    {"id": 29, "product_id": 4, "reviewer": "Neha J.", "rating": 5, "text": "Best product ever!!! Amazing amazing amazing must buy now!!!", "date": "2026-05-10"},
    {"id": 30, "product_id": 5, "reviewer": "Priya K.", "rating": 5, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-09"},
    {"id": 31, "product_id": 5, "reviewer": "Sanjay N.", "rating": 5, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-05-26"},
    {"id": 32, "product_id": 5, "reviewer": "Amit V.", "rating": 4, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-05-13"},
    {"id": 33, "product_id": 5, "reviewer": "Sneha R.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-05-11"},
    {"id": 34, "product_id": 5, "reviewer": "Vikram T.", "rating": 5, "text": "Perfect perfect perfect no issues at all totally worth it buy today", "date": "2026-05-04"},
    {"id": 35, "product_id": 5, "reviewer": "Sneha R.", "rating": 5, "text": "Superb quality superb price superb delivery best seller must buy", "date": "2026-05-04"},
    {"id": 36, "product_id": 5, "reviewer": "Rahul S.", "rating": 5, "text": "Best product ever!!! Amazing amazing amazing must buy now!!!", "date": "2026-05-04"},
    {"id": 37, "product_id": 6, "reviewer": "Pooja L.", "rating": 2, "text": "Quality is average, expected better given the price and brand claims.", "date": "2026-05-23"},
    {"id": 38, "product_id": 6, "reviewer": "Ritu W.", "rating": 5, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-30"},
    {"id": 39, "product_id": 6, "reviewer": "Arjun B.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-07"},
    {"id": 40, "product_id": 6, "reviewer": "Rahul S.", "rating": 1, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-06-03"},
    {"id": 41, "product_id": 6, "reviewer": "Rahul S.", "rating": 5, "text": "Perfect perfect perfect no issues at all totally worth it buy today", "date": "2026-06-03"},
    {"id": 42, "product_id": 6, "reviewer": "Neha J.", "rating": 5, "text": "Best product ever!!! Amazing amazing amazing must buy now!!!", "date": "2026-06-03"},
    {"id": 43, "product_id": 6, "reviewer": "Rohan G.", "rating": 5, "text": "Perfect perfect perfect no issues at all totally worth it buy today", "date": "2026-06-03"},
    {"id": 44, "product_id": 6, "reviewer": "Pooja L.", "rating": 5, "text": "Excellent excellent excellent 5 star 5 star best in class product buy now", "date": "2026-06-03"},
    {"id": 45, "product_id": 7, "reviewer": "Divya P.", "rating": 2, "text": "It's okay but not worth the price, there are better alternatives available.", "date": "2026-06-10"},
    {"id": 46, "product_id": 7, "reviewer": "Sanjay N.", "rating": 1, "text": "Quality is average, expected better given the price and brand claims.", "date": "2026-06-22"},
    {"id": 47, "product_id": 7, "reviewer": "Ritu W.", "rating": 1, "text": "Quality is average, expected better given the price and brand claims.", "date": "2026-06-03"},
    {"id": 48, "product_id": 7, "reviewer": "Rohan G.", "rating": 4, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-06-20"},
    {"id": 49, "product_id": 7, "reviewer": "Pooja L.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-29"},
    {"id": 50, "product_id": 8, "reviewer": "Karan D.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-31"},
    {"id": 51, "product_id": 8, "reviewer": "Pooja L.", "rating": 4, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-31"},
    {"id": 52, "product_id": 8, "reviewer": "Karan D.", "rating": 1, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-05-25"},
    {"id": 53, "product_id": 8, "reviewer": "Arjun B.", "rating": 5, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-06-20"},
    {"id": 54, "product_id": 8, "reviewer": "Amit V.", "rating": 5, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-05-26"},
    {"id": 55, "product_id": 9, "reviewer": "Manish C.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-05-10"},
    {"id": 56, "product_id": 9, "reviewer": "Ritu W.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-06-09"},
    {"id": 57, "product_id": 9, "reviewer": "Vikram T.", "rating": 1, "text": "Size/fit was off compared to the description, had to return it.", "date": "2026-06-05"},
    {"id": 58, "product_id": 9, "reviewer": "Karan D.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-28"},
    {"id": 59, "product_id": 10, "reviewer": "Divya P.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-06-02"},
    {"id": 60, "product_id": 10, "reviewer": "Kavya H.", "rating": 2, "text": "Size/fit was off compared to the description, had to return it.", "date": "2026-06-04"},
    {"id": 61, "product_id": 10, "reviewer": "Pooja L.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-06-17"},
    {"id": 62, "product_id": 10, "reviewer": "Kavya H.", "rating": 3, "text": "It works, though I think there are similar products for slightly less.", "date": "2026-06-07"},
    {"id": 63, "product_id": 11, "reviewer": "Anjali M.", "rating": 5, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-06-19"},
    {"id": 64, "product_id": 11, "reviewer": "Ritu W.", "rating": 1, "text": "Quality is average, expected better given the price and brand claims.", "date": "2026-05-10"},
    {"id": 65, "product_id": 11, "reviewer": "Ritu W.", "rating": 1, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-05-21"},
    {"id": 66, "product_id": 11, "reviewer": "Karan D.", "rating": 1, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-05-16"},
    {"id": 67, "product_id": 11, "reviewer": "Rahul S.", "rating": 5, "text": "Nice product", "date": "2026-06-19"},
    {"id": 68, "product_id": 11, "reviewer": "Manish C.", "rating": 5, "text": "Nice", "date": "2026-06-19"},
    {"id": 69, "product_id": 11, "reviewer": "Karan D.", "rating": 5, "text": "Good product nice", "date": "2026-06-19"},
    {"id": 70, "product_id": 12, "reviewer": "Rohan G.", "rating": 4, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-06-14"},
    {"id": 71, "product_id": 12, "reviewer": "Manish C.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-06-23"},
    {"id": 72, "product_id": 12, "reviewer": "Arjun B.", "rating": 5, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-29"},
    {"id": 73, "product_id": 12, "reviewer": "Karan D.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-05"},
    {"id": 74, "product_id": 12, "reviewer": "Pooja L.", "rating": 1, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-06-15"},
    {"id": 75, "product_id": 12, "reviewer": "Sneha R.", "rating": 5, "text": "Awesome product awesome price buy buy buy limited stock hurry", "date": "2026-05-09"},
    {"id": 76, "product_id": 12, "reviewer": "Ritu W.", "rating": 5, "text": "Excellent excellent excellent 5 star 5 star best in class product buy now", "date": "2026-05-09"},
    {"id": 77, "product_id": 12, "reviewer": "Divya P.", "rating": 5, "text": "Excellent excellent excellent 5 star 5 star best in class product buy now", "date": "2026-05-09"},
    {"id": 78, "product_id": 12, "reviewer": "Kavya H.", "rating": 5, "text": "Perfect perfect perfect no issues at all totally worth it buy today", "date": "2026-05-09"},
    {"id": 79, "product_id": 13, "reviewer": "Amit V.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-21"},
    {"id": 80, "product_id": 13, "reviewer": "Manish C.", "rating": 2, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-06-05"},
    {"id": 81, "product_id": 13, "reviewer": "Neha J.", "rating": 5, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-05-22"},
    {"id": 82, "product_id": 13, "reviewer": "Divya P.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-06-20"},
    {"id": 83, "product_id": 13, "reviewer": "Rohan G.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-05-06"},
    {"id": 84, "product_id": 13, "reviewer": "Vikram T.", "rating": 5, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-06-18"},
    {"id": 85, "product_id": 14, "reviewer": "Rohan G.", "rating": 5, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-06"},
    {"id": 86, "product_id": 14, "reviewer": "Rohan G.", "rating": 5, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-05-05"},
    {"id": 87, "product_id": 14, "reviewer": "Rohan G.", "rating": 3, "text": "It works, though I think there are similar products for slightly less.", "date": "2026-05-06"},
    {"id": 88, "product_id": 14, "reviewer": "Sneha R.", "rating": 4, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-17"},
    {"id": 89, "product_id": 14, "reviewer": "Vikram T.", "rating": 5, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-18"},
    {"id": 90, "product_id": 14, "reviewer": "Anjali M.", "rating": 5, "text": "Superb quality superb price superb delivery best seller must buy", "date": "2026-06-15"},
    {"id": 91, "product_id": 14, "reviewer": "Priya K.", "rating": 5, "text": "Excellent excellent excellent 5 star 5 star best in class product buy now", "date": "2026-06-15"},
    {"id": 92, "product_id": 14, "reviewer": "Karan D.", "rating": 5, "text": "Superb quality superb price superb delivery best seller must buy", "date": "2026-06-15"},
    {"id": 93, "product_id": 15, "reviewer": "Anjali M.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-29"},
    {"id": 94, "product_id": 15, "reviewer": "Rahul S.", "rating": 5, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-05-03"},
    {"id": 95, "product_id": 15, "reviewer": "Manish C.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-16"},
    {"id": 96, "product_id": 15, "reviewer": "Sanjay N.", "rating": 5, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-06-04"},
    {"id": 97, "product_id": 15, "reviewer": "Karan D.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-06-14"},
    {"id": 98, "product_id": 16, "reviewer": "Priya K.", "rating": 5, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-05-23"},
    {"id": 99, "product_id": 16, "reviewer": "Rohan G.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-05-05"},
    {"id": 100, "product_id": 16, "reviewer": "Sanjay N.", "rating": 4, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-06-12"},
    {"id": 101, "product_id": 16, "reviewer": "Divya P.", "rating": 3, "text": "It works, though I think there are similar products for slightly less.", "date": "2026-05-19"},
    {"id": 102, "product_id": 16, "reviewer": "Anjali M.", "rating": 2, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-05-12"},
    {"id": 103, "product_id": 16, "reviewer": "Arjun B.", "rating": 5, "text": "Osm product", "date": "2026-05-01"},
    {"id": 104, "product_id": 16, "reviewer": "Arjun B.", "rating": 5, "text": "Good", "date": "2026-05-01"},
    {"id": 105, "product_id": 16, "reviewer": "Priya K.", "rating": 5, "text": "Nice", "date": "2026-05-01"},
    {"id": 106, "product_id": 16, "reviewer": "Karan D.", "rating": 5, "text": "Nice", "date": "2026-05-01"},
    {"id": 107, "product_id": 17, "reviewer": "Rohan G.", "rating": 4, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-05-31"},
    {"id": 108, "product_id": 17, "reviewer": "Rahul S.", "rating": 4, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-06-02"},
    {"id": 109, "product_id": 17, "reviewer": "Priya K.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-26"},
    {"id": 110, "product_id": 17, "reviewer": "Divya P.", "rating": 5, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-06-10"},
    {"id": 111, "product_id": 17, "reviewer": "Arjun B.", "rating": 5, "text": "Good product nice", "date": "2026-06-03"},
    {"id": 112, "product_id": 17, "reviewer": "Ritu W.", "rating": 5, "text": "Good", "date": "2026-06-03"},
    {"id": 113, "product_id": 17, "reviewer": "Neha J.", "rating": 5, "text": "Good product nice", "date": "2026-06-03"},
    {"id": 114, "product_id": 18, "reviewer": "Rahul S.", "rating": 4, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-06-03"},
    {"id": 115, "product_id": 18, "reviewer": "Divya P.", "rating": 3, "text": "It works, though I think there are similar products for slightly less.", "date": "2026-06-21"},
    {"id": 116, "product_id": 18, "reviewer": "Pooja L.", "rating": 4, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-06-10"},
    {"id": 117, "product_id": 18, "reviewer": "Manish C.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-06-23"},
    {"id": 118, "product_id": 19, "reviewer": "Rahul S.", "rating": 2, "text": "Quality is average, expected better given the price and brand claims.", "date": "2026-05-17"},
    {"id": 119, "product_id": 19, "reviewer": "Amit V.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-06-12"},
    {"id": 120, "product_id": 19, "reviewer": "Amit V.", "rating": 2, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-06-21"},
    {"id": 121, "product_id": 20, "reviewer": "Manish C.", "rating": 1, "text": "Quality is average, expected better given the price and brand claims.", "date": "2026-06-17"},
    {"id": 122, "product_id": 20, "reviewer": "Neha J.", "rating": 4, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-05-31"},
    {"id": 123, "product_id": 20, "reviewer": "Vikram T.", "rating": 1, "text": "It's okay but not worth the price, there are better alternatives available.", "date": "2026-05-05"},
    {"id": 124, "product_id": 20, "reviewer": "Ritu W.", "rating": 4, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-05-01"},
    {"id": 125, "product_id": 20, "reviewer": "Karan D.", "rating": 5, "text": "Awesome product awesome price buy buy buy limited stock hurry", "date": "2026-05-18"},
    {"id": 126, "product_id": 20, "reviewer": "Neha J.", "rating": 5, "text": "Perfect perfect perfect no issues at all totally worth it buy today", "date": "2026-05-18"},
    {"id": 127, "product_id": 20, "reviewer": "Neha J.", "rating": 5, "text": "Awesome product awesome price buy buy buy limited stock hurry", "date": "2026-05-18"},
    {"id": 128, "product_id": 20, "reviewer": "Manish C.", "rating": 5, "text": "Awesome product awesome price buy buy buy limited stock hurry", "date": "2026-05-18"},
    {"id": 129, "product_id": 20, "reviewer": "Sneha R.", "rating": 5, "text": "Perfect perfect perfect no issues at all totally worth it buy today", "date": "2026-05-18"},
    {"id": 130, "product_id": 21, "reviewer": "Neha J.", "rating": 5, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-02"},
    {"id": 131, "product_id": 21, "reviewer": "Karan D.", "rating": 5, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-25"},
    {"id": 132, "product_id": 21, "reviewer": "Amit V.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-05-05"},
    {"id": 133, "product_id": 21, "reviewer": "Rohan G.", "rating": 5, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-05-09"},
    {"id": 134, "product_id": 22, "reviewer": "Anjali M.", "rating": 5, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-02"},
    {"id": 135, "product_id": 22, "reviewer": "Neha J.", "rating": 5, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-26"},
    {"id": 136, "product_id": 22, "reviewer": "Arjun B.", "rating": 2, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-05-25"},
    {"id": 137, "product_id": 22, "reviewer": "Arjun B.", "rating": 4, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-21"},
    {"id": 138, "product_id": 22, "reviewer": "Rahul S.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-05-13"},
    {"id": 139, "product_id": 23, "reviewer": "Amit V.", "rating": 5, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-25"},
    {"id": 140, "product_id": 23, "reviewer": "Priya K.", "rating": 5, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-06-24"},
    {"id": 141, "product_id": 23, "reviewer": "Vikram T.", "rating": 5, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-06-10"},
    {"id": 142, "product_id": 23, "reviewer": "Arjun B.", "rating": 5, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-06-02"},
    {"id": 143, "product_id": 23, "reviewer": "Sanjay N.", "rating": 5, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-02"},
    {"id": 144, "product_id": 24, "reviewer": "Vikram T.", "rating": 2, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-05-29"},
    {"id": 145, "product_id": 24, "reviewer": "Vikram T.", "rating": 2, "text": "Size/fit was off compared to the description, had to return it.", "date": "2026-05-04"},
    {"id": 146, "product_id": 24, "reviewer": "Neha J.", "rating": 5, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-05-19"},
    {"id": 147, "product_id": 24, "reviewer": "Divya P.", "rating": 5, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-06-11"},
    {"id": 148, "product_id": 24, "reviewer": "Sneha R.", "rating": 5, "text": "Good", "date": "2026-06-05"},
    {"id": 149, "product_id": 24, "reviewer": "Anjali M.", "rating": 5, "text": "Nice product", "date": "2026-06-05"},
    {"id": 150, "product_id": 24, "reviewer": "Karan D.", "rating": 5, "text": "Osm product", "date": "2026-06-05"},
    {"id": 151, "product_id": 24, "reviewer": "Ritu W.", "rating": 5, "text": "Osm product", "date": "2026-06-05"},
    {"id": 152, "product_id": 24, "reviewer": "Divya P.", "rating": 5, "text": "Very nice product good quality", "date": "2026-06-05"},
    {"id": 153, "product_id": 25, "reviewer": "Vikram T.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-05-28"},
    {"id": 154, "product_id": 25, "reviewer": "Arjun B.", "rating": 4, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-12"},
    {"id": 155, "product_id": 25, "reviewer": "Rohan G.", "rating": 4, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-24"},
    {"id": 156, "product_id": 25, "reviewer": "Kavya H.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-05-02"},
    {"id": 157, "product_id": 25, "reviewer": "Arjun B.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-18"},
    {"id": 158, "product_id": 26, "reviewer": "Karan D.", "rating": 4, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-06-13"},
    {"id": 159, "product_id": 26, "reviewer": "Manish C.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-26"},
    {"id": 160, "product_id": 26, "reviewer": "Priya K.", "rating": 4, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-05-09"},
    {"id": 161, "product_id": 26, "reviewer": "Amit V.", "rating": 5, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-01"},
    {"id": 162, "product_id": 26, "reviewer": "Sneha R.", "rating": 5, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-16"},
    {"id": 163, "product_id": 26, "reviewer": "Manish C.", "rating": 4, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-06-22"},
    {"id": 164, "product_id": 26, "reviewer": "Vikram T.", "rating": 5, "text": "Excellent excellent excellent 5 star 5 star best in class product buy now", "date": "2026-06-19"},
    {"id": 165, "product_id": 26, "reviewer": "Priya K.", "rating": 5, "text": "Superb quality superb price superb delivery best seller must buy", "date": "2026-06-19"},
    {"id": 166, "product_id": 26, "reviewer": "Vikram T.", "rating": 5, "text": "Superb quality superb price superb delivery best seller must buy", "date": "2026-06-19"},
    {"id": 167, "product_id": 27, "reviewer": "Neha J.", "rating": 1, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-05-05"},
    {"id": 168, "product_id": 27, "reviewer": "Divya P.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-17"},
    {"id": 169, "product_id": 27, "reviewer": "Neha J.", "rating": 1, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-06-04"},
    {"id": 170, "product_id": 27, "reviewer": "Divya P.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-05-21"},
    {"id": 171, "product_id": 27, "reviewer": "Kavya H.", "rating": 4, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-02"},
    {"id": 172, "product_id": 27, "reviewer": "Karan D.", "rating": 1, "text": "Size/fit was off compared to the description, had to return it.", "date": "2026-05-02"},
    {"id": 173, "product_id": 28, "reviewer": "Divya P.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-24"},
    {"id": 174, "product_id": 28, "reviewer": "Sanjay N.", "rating": 5, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-24"},
    {"id": 175, "product_id": 28, "reviewer": "Ritu W.", "rating": 4, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-05-14"},
    {"id": 176, "product_id": 28, "reviewer": "Karan D.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-06-19"},
    {"id": 177, "product_id": 28, "reviewer": "Neha J.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-06-18"},
    {"id": 178, "product_id": 28, "reviewer": "Ritu W.", "rating": 4, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-15"},
    {"id": 179, "product_id": 29, "reviewer": "Priya K.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-05-26"},
    {"id": 180, "product_id": 29, "reviewer": "Priya K.", "rating": 5, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-05-04"},
    {"id": 181, "product_id": 29, "reviewer": "Sneha R.", "rating": 5, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-06-16"},
    {"id": 182, "product_id": 30, "reviewer": "Neha J.", "rating": 5, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-05-03"},
    {"id": 183, "product_id": 30, "reviewer": "Manish C.", "rating": 2, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-05-22"},
    {"id": 184, "product_id": 30, "reviewer": "Amit V.", "rating": 4, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-05-18"},
    {"id": 185, "product_id": 30, "reviewer": "Pooja L.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-25"},
    {"id": 186, "product_id": 31, "reviewer": "Priya K.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-05-06"},
    {"id": 187, "product_id": 31, "reviewer": "Manish C.", "rating": 2, "text": "Quality is average, expected better given the price and brand claims.", "date": "2026-06-04"},
    {"id": 188, "product_id": 31, "reviewer": "Kavya H.", "rating": 5, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-02"},
    {"id": 189, "product_id": 31, "reviewer": "Priya K.", "rating": 4, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-05-25"},
    {"id": 190, "product_id": 31, "reviewer": "Amit V.", "rating": 5, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-05-13"},
    {"id": 191, "product_id": 32, "reviewer": "Arjun B.", "rating": 4, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-17"},
    {"id": 192, "product_id": 32, "reviewer": "Amit V.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-05-01"},
    {"id": 193, "product_id": 32, "reviewer": "Manish C.", "rating": 4, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-31"},
    {"id": 194, "product_id": 32, "reviewer": "Rohan G.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-06-20"},
    {"id": 195, "product_id": 32, "reviewer": "Ritu W.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-05-09"},
    {"id": 196, "product_id": 32, "reviewer": "Vikram T.", "rating": 5, "text": "Osm product", "date": "2026-06-21"},
    {"id": 197, "product_id": 32, "reviewer": "Divya P.", "rating": 5, "text": "Nice", "date": "2026-06-21"},
    {"id": 198, "product_id": 32, "reviewer": "Arjun B.", "rating": 5, "text": "Very nice product good quality", "date": "2026-06-21"},
    {"id": 199, "product_id": 32, "reviewer": "Pooja L.", "rating": 5, "text": "Osm product", "date": "2026-06-21"},
    {"id": 200, "product_id": 32, "reviewer": "Amit V.", "rating": 5, "text": "Osm product", "date": "2026-06-21"},
    {"id": 201, "product_id": 33, "reviewer": "Amit V.", "rating": 4, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-05-27"},
    {"id": 202, "product_id": 33, "reviewer": "Kavya H.", "rating": 2, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-05-11"},
    {"id": 203, "product_id": 33, "reviewer": "Amit V.", "rating": 3, "text": "Product is fine, nothing exceptional but does what it's supposed to.", "date": "2026-05-17"},
    {"id": 204, "product_id": 33, "reviewer": "Manish C.", "rating": 5, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-06-15"},
    {"id": 205, "product_id": 33, "reviewer": "Divya P.", "rating": 5, "text": "Good product nice", "date": "2026-05-09"},
    {"id": 206, "product_id": 33, "reviewer": "Sneha R.", "rating": 5, "text": "Nice", "date": "2026-05-09"},
    {"id": 207, "product_id": 33, "reviewer": "Neha J.", "rating": 5, "text": "Nice", "date": "2026-05-09"},
    {"id": 208, "product_id": 33, "reviewer": "Rohan G.", "rating": 5, "text": "Nice", "date": "2026-05-09"},
    {"id": 209, "product_id": 34, "reviewer": "Anjali M.", "rating": 2, "text": "Quality is average, expected better given the price and brand claims.", "date": "2026-05-16"},
    {"id": 210, "product_id": 34, "reviewer": "Karan D.", "rating": 5, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-06-07"},
    {"id": 211, "product_id": 34, "reviewer": "Divya P.", "rating": 5, "text": "Not bad at all, comfortable to use and looks premium for the price point.", "date": "2026-05-16"},
    {"id": 212, "product_id": 34, "reviewer": "Sneha R.", "rating": 2, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-05-03"},
    {"id": 213, "product_id": 34, "reviewer": "Priya K.", "rating": 5, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-24"},
    {"id": 214, "product_id": 35, "reviewer": "Pooja L.", "rating": 4, "text": "Packaging was solid and product matches the description. Satisfied overall.", "date": "2026-05-05"},
    {"id": 215, "product_id": 35, "reviewer": "Rohan G.", "rating": 5, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-06-08"},
    {"id": 216, "product_id": 35, "reviewer": "Pooja L.", "rating": 1, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-06-10"},
    {"id": 217, "product_id": 35, "reviewer": "Priya K.", "rating": 5, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-10"},
    {"id": 218, "product_id": 35, "reviewer": "Rahul S.", "rating": 5, "text": "Superb quality superb price superb delivery best seller must buy", "date": "2026-05-17"},
    {"id": 219, "product_id": 35, "reviewer": "Kavya H.", "rating": 5, "text": "Superb quality superb price superb delivery best seller must buy", "date": "2026-05-17"},
    {"id": 220, "product_id": 35, "reviewer": "Anjali M.", "rating": 5, "text": "Perfect perfect perfect no issues at all totally worth it buy today", "date": "2026-05-17"},
    {"id": 221, "product_id": 36, "reviewer": "Ritu W.", "rating": 5, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-06-05"},
    {"id": 222, "product_id": 36, "reviewer": "Vikram T.", "rating": 5, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-06-12"},
    {"id": 223, "product_id": 36, "reviewer": "Rohan G.", "rating": 1, "text": "Product stopped working properly after a week of normal use, disappointed.", "date": "2026-05-26"},
    {"id": 224, "product_id": 36, "reviewer": "Priya K.", "rating": 5, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-05-27"},
    {"id": 225, "product_id": 36, "reviewer": "Rahul S.", "rating": 5, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-27"},
    {"id": 226, "product_id": 37, "reviewer": "Rahul S.", "rating": 2, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-05-14"},
    {"id": 227, "product_id": 37, "reviewer": "Amit V.", "rating": 5, "text": "Value for money product. Does the job well, would recommend to others.", "date": "2026-05-08"},
    {"id": 228, "product_id": 37, "reviewer": "Anjali M.", "rating": 5, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-06-19"},
    {"id": 229, "product_id": 37, "reviewer": "Sanjay N.", "rating": 4, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-06-11"},
    {"id": 230, "product_id": 37, "reviewer": "Pooja L.", "rating": 4, "text": "Works well for daily use, battery/performance is better than I expected.", "date": "2026-05-10"},
    {"id": 231, "product_id": 37, "reviewer": "Sneha R.", "rating": 5, "text": "Awesome product awesome price buy buy buy limited stock hurry", "date": "2026-06-03"},
    {"id": 232, "product_id": 37, "reviewer": "Ritu W.", "rating": 5, "text": "Excellent excellent excellent 5 star 5 star best in class product buy now", "date": "2026-06-03"},
    {"id": 233, "product_id": 37, "reviewer": "Neha J.", "rating": 5, "text": "Excellent excellent excellent 5 star 5 star best in class product buy now", "date": "2026-06-03"},
    {"id": 234, "product_id": 38, "reviewer": "Priya K.", "rating": 3, "text": "Average experience, matches most of the description provided.", "date": "2026-05-21"},
    {"id": 235, "product_id": 38, "reviewer": "Anjali M.", "rating": 1, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-06-15"},
    {"id": 236, "product_id": 38, "reviewer": "Karan D.", "rating": 2, "text": "Quality is average, expected better given the price and brand claims.", "date": "2026-06-09"},
    {"id": 237, "product_id": 39, "reviewer": "Anjali M.", "rating": 5, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-06-03"},
    {"id": 238, "product_id": 39, "reviewer": "Karan D.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-16"},
    {"id": 239, "product_id": 39, "reviewer": "Sanjay N.", "rating": 5, "text": "Really happy with this purchase, works exactly as described and arrived on time.", "date": "2026-05-08"},
    {"id": 240, "product_id": 39, "reviewer": "Divya P.", "rating": 5, "text": "Decent quality, took a star off because delivery was slightly delayed.", "date": "2026-05-20"},
    {"id": 241, "product_id": 40, "reviewer": "Rahul S.", "rating": 4, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-02"},
    {"id": 242, "product_id": 40, "reviewer": "Manish C.", "rating": 2, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-05-16"},
    {"id": 243, "product_id": 40, "reviewer": "Ritu W.", "rating": 1, "text": "Delivery took much longer than promised and packaging was damaged.", "date": "2026-06-21"},
    {"id": 244, "product_id": 40, "reviewer": "Kavya H.", "rating": 4, "text": "Good build quality for the price. Been using it for two weeks, no issues so far.", "date": "2026-05-23"},
    {"id": 245, "product_id": 40, "reviewer": "Vikram T.", "rating": 4, "text": "Bought this after comparing a few options, glad I chose this one.", "date": "2026-05-03"},
    {"id": 246, "product_id": 40, "reviewer": "Priya K.", "rating": 5, "text": "Perfect perfect perfect no issues at all totally worth it buy today", "date": "2026-06-16"},
    {"id": 247, "product_id": 40, "reviewer": "Sanjay N.", "rating": 5, "text": "Excellent excellent excellent 5 star 5 star best in class product buy now", "date": "2026-06-16"},
    {"id": 248, "product_id": 40, "reviewer": "Rahul S.", "rating": 5, "text": "Best product ever!!! Amazing amazing amazing must buy now!!!", "date": "2026-06-16"},
    {"id": 249, "product_id": 40, "reviewer": "Sneha R.", "rating": 5, "text": "Excellent excellent excellent 5 star 5 star best in class product buy now", "date": "2026-06-16"},
]


class ProductLensEngine:
    def __init__(self):
        self.products = PRODUCTS_DATA
        self.reviews = REVIEWS_DATA
        
        self.products_df = pd.DataFrame(self.products)
        self.reviews_df = pd.DataFrame(self.reviews)

        self._build_recommender()

    # ---------- Recommendation engine (content-based, TF-IDF + cosine similarity) ----------
    def _build_recommender(self):
        corpus = (self.products_df["name"] + " " + self.products_df["category"] + " " + self.products_df["description"])
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def get_recommendations(self, product_id, top_n=4):
        idx_series = self.products_df.index[self.products_df["id"] == product_id]
        if len(idx_series) == 0:
            return []
        idx = idx_series[0]

        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        recommendations = []
        for i, score in scores:
            row = self.products_df.iloc[i]
            recommendations.append({
                "id": int(row["id"]),
                "name": row["name"],
                "category": row["category"],
                "price": int(row["price"]),
                "similarity": round(float(score), 3),
            })
        return recommendations

    # ---------- Fake / spam review detection (rule-based scoring, no external labels needed) ----------
    def analyze_reviews(self, product_id):
        product_reviews = self.reviews_df[self.reviews_df["product_id"] == product_id].copy()
        if product_reviews.empty:
            return []

        # Signal 1: same-day burst — many reviews posted the same date is a spam signal
        date_counts = product_reviews["date"].value_counts()
        burst_dates = set(date_counts[date_counts >= 3].index)

        # Signal 2: near-duplicate / generic text reused across reviewers
        text_counts = Counter(product_reviews["text"].str.lower().str.strip())

        # Signal 3: excessive punctuation / hype words
        hype_words = {"amazing", "best", "awesome", "superb", "perfect", "excellent", "buy now", "hurry", "must buy"}

        results = []
        for _, r in product_reviews.iterrows():
            score = 0
            reasons = []

            if r["date"] in burst_dates:
                score += 2
                reasons.append("posted in a same-day review burst")

            if text_counts[r["text"].lower().strip()] >= 3:
                score += 2
                reasons.append("near-identical text reused across multiple reviews")

            text_lower = r["text"].lower()
            hype_hits = sum(1 for w in hype_words if w in text_lower)
            if hype_hits >= 2:
                score += 2
                reasons.append("excessive hype/marketing language")

            if len(r["text"].split()) <= 3 and r["rating"] == 5:
                score += 1
                reasons.append("very short 5-star text with no detail")

            if text_lower.count("!") >= 2:
                score += 1
                reasons.append("excessive exclamation marks")

            is_suspicious = score >= 3
            sentiment = self._simple_sentiment(r["text"], r["rating"])

            results.append({
                "id": int(r["id"]),
                "reviewer": r["reviewer"],
                "rating": int(r["rating"]),
                "text": r["text"],
                "date": r["date"],
                "suspicion_score": score,
                "is_suspicious": bool(is_suspicious),
                "reasons": reasons,
                "sentiment": sentiment,
            })

        results.sort(key=lambda x: x["suspicion_score"], reverse=True)
        return results

    def _simple_sentiment(self, text, rating):
        positive_words = {"good", "great", "excellent", "happy", "satisfied", "recommend", "value", "well", "nice", "glad", "better"}
        negative_words = {"disappointed", "average", "delayed", "damaged", "stopped", "return", "off", "worse", "issue"}

        text_lower = text.lower()
        pos_hits = sum(1 for w in positive_words if w in text_lower)
        neg_hits = sum(1 for w in negative_words if w in text_lower)

        if pos_hits > neg_hits and rating >= 4:
            return "positive"
        elif neg_hits > pos_hits or rating <= 2:
            return "negative"
        return "neutral"

    # ---------- Aggregate stats for a product ----------
    def get_review_summary(self, product_id):
        analyzed = self.analyze_reviews(product_id)
        if not analyzed:
            return {"total": 0, "suspicious_count": 0, "avg_rating": 0, "sentiment_breakdown": {}}

        total = len(analyzed)
        suspicious_count = sum(1 for r in analyzed if r["is_suspicious"])
        avg_rating = round(sum(r["rating"] for r in analyzed) / total, 2)

        sentiment_breakdown = Counter(r["sentiment"] for r in analyzed)

        # rating average excluding suspicious reviews — a more "honest" score
        genuine = [r for r in analyzed if not r["is_suspicious"]]
        genuine_avg = round(sum(r["rating"] for r in genuine) / len(genuine), 2) if genuine else avg_rating

        return {
            "total": total,
            "suspicious_count": suspicious_count,
            "avg_rating": avg_rating,
            "genuine_avg_rating": genuine_avg,
            "sentiment_breakdown": dict(sentiment_breakdown),
        }

    def get_all_products(self, category=None, search=None):
        df = self.products_df
        if category:
            df = df[df["category"] == category]
        if search:
            mask = df["name"].str.lower().str.contains(search.lower()) | df["description"].str.lower().str.contains(search.lower())
            df = df[mask]
        return df.to_dict(orient="records")

    def get_product(self, product_id):
        row = self.products_df[self.products_df["id"] == product_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_categories(self):
        return sorted(self.products_df["category"].unique().tolist())
