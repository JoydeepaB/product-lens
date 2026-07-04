import os
from flask import Flask, render_template, request, jsonify
from engine import ProductLensEngine

app = Flask(__name__, template_folder='.')
engine = ProductLensEngine()


@app.route("/")
def home():
    category = request.args.get("category")
    search = request.args.get("q")
    products = engine.get_all_products(category=category, search=search)
    categories = engine.get_categories()
    return render_template("index.html", products=products, categories=categories,
                            selected_category=category, search_query=search or "")


@app.route("/product/<int:product_id>")
def product_detail(product_id):
    product = engine.get_product(product_id)
    if not product:
        return "Product not found", 404

    recommendations = engine.get_recommendations(product_id)
    reviews = engine.analyze_reviews(product_id)
    summary = engine.get_review_summary(product_id)

    return render_template("product.html", product=product, recommendations=recommendations,
                            reviews=reviews, summary=summary)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
