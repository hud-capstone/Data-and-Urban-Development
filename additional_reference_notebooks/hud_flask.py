from flask import Flask, redirect, url_for, render_template

# create instance of web app

app = Flask(__name__)

# define the pages that will be on the site
@app.route("/<name>") #format is 'app.route(path)'.  '/' is the default page
def home(name):
    """
    Return HTML
    """
    return render_template("index.html")

# @app.route("/Alec") #format is 'app.route(path)'.  '/' is the default page
# def alec():
#     """
#     Return Alec
#     """
#     return render_template("Alec.html")

# @app.route("/Daniel") #format is 'app.route(path)'.  '/' is the default page
# def daniel():
#     """
#     Return Daniel
#     """
#     return render_template("Daniel.html")

# @app.route("/Noah") #format is 'app.route(path)'.  '/' is the default page
# def noah():
#     """
#     Return Noah
#     """
#     return render_template("Noah.html")

# @app.route("/Nick") #format is 'app.route(path)'.  '/' is the default page
# def nick():
#     """
#     Return Nick
#     """
#     return render_template("Nick.html")

# @app.route("/<name>") # if not happy, try ("/<nick>")
# def visitor(name):
#     """
#     Grabs the HTML in <>, uses it to fill in the parameter, and returns it
#     """
#     return f"Aloha, {name}!"

# @app.route("/admin/")
# def admin):
#     """
#     Redirects to unauthorized user
#     """
#     return redirect(url_for("user", name="Admin!")) 

# run the website

if __name__ == "__main__":
    app.run(debug=True) # keeps us from having to re-run the server every time we make a change; changes are automatically detected
