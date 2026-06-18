# Challenge 1: Deploy an API in Azure

## Overview

🎯 **Exercise objectives**:

* Build a Python package ready for deployment.
* Deploy a FastAPI API to Azure **using the Azure CLI**, understanding each resource you create.
* Test your code with pytest.
* **(Bonus)** Automate the deployment with CI/CD using GitHub Actions.

🔧 **Tools**:

* **VS Code** for interactive coding
* **FastAPI** to build REST APIs
* **joblib** to save/load predictive models
* **Azure CLI (`az`)** to create and manage cloud resources from the terminal
* **Azure Portal** to visualize and understand what you created
* **GitHub** to host the code of your Apps

---
## 📃 Instructions

In this challenge, you will deploy a machine learning model that predicts wine quality, exposed through a FastAPI API. You will create every Azure resource **from the command line**, and use the Azure Portal to *verify and understand* what each command did.

### 0. How Azure is organized (read this first)

Before touching anything, here is the mental model of Azure — three nested levels:

```
Subscription (AI_Integration)   ← the billing boundary: everything you create is charged here
└── Resource Group              ← a logical folder that groups related resources
    └── Resources               ← your actual things: an API, a database, a model...
```

- A **Subscription** is the account where usage is billed. You all share the `AI_Integration` subscription.
- A **Resource Group (RG)** is a folder that holds related resources so you can manage — and delete — them together.
- A **Resource** is an individual service: a web app, a storage account, an AI model, etc.

👉 **We already created one Resource Group for each of you**, named with **your first name in lowercase, without accents** (if your name is `Frédéric`, your RG is `frederic`). You will create all your resources inside *your* RG. You have **Contributor** rights there — meaning you can create, modify and delete resources in your own folder, but not in anyone else's.

> 💡 This is exactly how it works in a real company: a platform team gives you a resource group to work in, and you build inside it.

---
### 1. Explore your API and test it locally

- Open `VS Code` with the `fastapi-env` environment and open the exercise folder.

- Open the file `main.py` and verify that the code loads the model correctly.

- Check that you have the right features and types in the endpoint to make the inference. Note the `pydantic` Model used to process the features.

- Go to Environments in the Anaconda Navigator and click the green play icon (▶️) next to your `fastapi-env` environment.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-21 171227.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBMmFIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--181032820c2d01d6b445b339f1db45993b80b2d1/Capture%20d'%C3%A9cran%202024-09-21%20171227.png" />
</figure>

- A terminal will open with your environment activated. Launch the API server:

  ```bash
  cd # Gets back to the $HOME directory
  cd Documents/GitHub/ai-integration-se-challenges/02-AI-Integration-in-Software/01-Deploy-API-in-Azure
  fastapi dev main.py
  ```

- Open your browser at the localhost address shown in your terminal: `http://127.0.0.1:8000/docs`. This is your API's documentation page. Open the predict endpoint, click `Try it out`, enter some feature values and execute — you should get a prediction.

- Now try the API with code: open `test.py` in VS Code (with the fastapi-env) and run it. You should get a predicted value and probability.

- If you could open the FastAPI interface and make predictions, you've successfully exposed your model.

- Close your browser. In the terminal, press `Ctrl + C` to stop FastAPI. Close the terminal.

---
### 2. Build your local Python package

- Create a new local repository for your API. For instance at `Documents/GitHub/yourwineapi`. Use a unique name without spaces — your first name works well, e.g. `frederic-wineapi`.

- Use GitHub Desktop to initialize it with Git: `Create new repository`, select the folder location, give it the same name as your folder.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 183118.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM2lIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--94d3217bf9f08e3bd9520f686723620c6709446e/Capture%20d'%C3%A9cran%202024-09-22%20183118.png" />
</figure>

<figure style="width: 60%">
  <img alt="Capture d'écran 2024-09-22 183210.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM21IQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--bbe47a39b1d12ef05940f435ebe5b1fec1a4516b/Capture%20d'%C3%A9cran%202024-09-22%20183210.png" />
</figure>

- Copy `main.py` and the model `lr.joblib` into the new repository. Commit your changes in GitHub Desktop with a message (e.g. `First commit`).

- Create a `requirements.txt` with the libraries needed to run your API:

  ```text
  pip>=9
  setuptools>=26
  wheel>=0.29
  pandas
  pytest
  coverage
  numpy
  httpx
  scikit-learn==1.7.2
  joblib
  memoized-property
  gcsfs
  termcolor
  uvicorn
  gunicorn
  fastapi
  pydantic
  ```

  > ⚠️ Note the **pinned scikit-learn version**: the model `lr.joblib` was trained with this exact version. A joblib model is tied to the library version that produced it — pinning guarantees your tests and your deployed API load the model correctly. `gunicorn` is the production server Azure will use to launch the API.

  > 💡 That's all the app needs. In many tutorials you'd also see a `setup.py` or a `setup.sh` start script — we don't need them: Azure installs your `requirements.txt`, and we'll give it the start command directly in the Web App configuration (you'll see where).

- Your repository `Documents/GitHub/yourwineapi` should contain:

  ```
  yourwineapi
  │   lr.joblib
  │   main.py
  │   requirements.txt
  ```

- Save and commit this checkpoint in GitHub Desktop.

---
### 3. Build and test your code with pytest

In MLOps, testing code enables test automation for CI/CD. We'll evaluate locally with `pytest` first, then activate CI/CD with GitHub Actions later (bonus).

- Create a folder `tests` with an empty `__init__.py` and a file `test_fast.py`. From the root of `yourwineapi`:

  ```bash
  # macOS / Linux
  mkdir tests
  touch tests/__init__.py tests/test_fast.py
  ```

  ```cmd
  # Windows (Anaconda Prompt / cmd)
  mkdir tests
  type nul > tests\__init__.py
  type nul > tests\test_fast.py
  ```

  > 💡 The empty `__init__.py` lets pytest resolve imports from the project root, so `from main import app` works.

- In `test_fast.py`, add:

  ```python
  from fastapi.testclient import TestClient
  from main import app
  import json
  import pytest

  # Initialize a FastAPI client
  client = TestClient(app)

  # Test your root endpoint
  def test_read_main():
      response = client.get("/")
      assert response.status_code == 200
      assert response.json() == {"message": "Hello stranger! This API allow you to evaluate the quality of red wine. Go to the /docs for more details."}

  # Test your predict endpoint
  def test_predict():
      response = client.post(
          "/predict",
          headers={'accept': 'application/json', 'Content-Type': 'application/json'},
          json={'alcohol': 9.4, 'volatile_acidity': 0.7},
      )
      assert response.status_code == 200
      result = json.loads(response.json())
      assert result['prediction'] == 0
      assert result['probability'] == pytest.approx([0.7759, 0.2241], abs=1e-3)
  ```

  > 🧠 **Why `pytest.approx`?** Model probabilities are floats whose last decimals can shift with the library version or the machine. In MLOps you never assert float predictions with strict equality — you test within a tolerance. An exact comparison would break for the wrong reasons.

- Open an Anaconda Prompt with the `fastapi-env`. From the **project root** (not from inside `tests/`):

  ```bash
  cd # Gets back to the $HOME directory
  cd Documents/GitHub/yourwineapi
  pytest tests/test_fast.py
  ```

You should get **2 passed**. You may also see a harmless `UserWarning: X does not have valid feature names` (the model was trained on a DataFrame, the API predicts on a numpy array). If pytest is not found:

  ```bash
  pip install pytest
  pytest tests/test_fast.py
  ```

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 190754.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM3FIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--91795943f3e6645cf28d6fdb0cb6b74aa85a6eb8/Capture%20d'%C3%A9cran%202024-09-22%20190754.png" />
</figure>

If you do not obtain this result, ask a TA for help.

- Create a `.gitignore` file:

  ```text
  .DS_Store
  __pycache__
  .ipynb_checkpoints
  .pytest_cache
  ```

- Your repository should now contain:

  ```
  yourwineapi
  │   lr.joblib
  │   main.py
  │   requirements.txt
  │   .gitignore
  │
  └───tests
  │   │   __init__.py
  │   │   test_fast.py
  ```

---
### 4. Create a GitHub repository and link it to your local project

- In GitHub Desktop, from your local repository, use **Publish repository** to push it to GitHub. Follow the instructions.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 194206.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM3VIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--f2aa3222fa325c92849f987f38771178c83394c2/Capture%20d'%C3%A9cran%202024-09-22%20194206.png" />
</figure>

<figure style="width: 60%">
  <img alt="Capture d'écran 2024-09-22 194510.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM3lIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--c414b42973f16c250dc3e46135e86cef0e0896cf/Capture%20d'%C3%A9cran%202024-09-22%20194510.png" />
</figure>

- Open your GitHub account and check the remote repository has all your files. If not, ask a TA for help.

---
### 5. Connect to Azure with the CLI

- Go to [Azure Portal](https://portal.azure.com) and connect with the credentials that Le Wagon provided you during the setup.

- Check that you have access to the `AI_Integration` Subscription 🔑.

From here on, we create all cloud resources from the terminal with the **Azure CLI** (`az`). This is how MLOps/platform engineers work in real life: the portal is great to *explore and understand*, but resources are created with code so the process is **reproducible**.

- **Check the Azure CLI is installed:**

  ```bash
  az --version
  ```

  If the command is not found, ask a TA (it should have been installed during setup).

- **Log in:**

  ```bash
  az login
  ```

  🧠 This opens a browser to authenticate you and stores a token locally — every following `az` command acts on your behalf.

- **Select the shared subscription:**

  ```bash
  az account set --subscription "AI_Integration"
  az account show --output table
  ```

  🔑 If you don't see the `AI_Integration` subscription, ask for help.

---
### 6. Create your resources with the Azure CLI

We'll create the **App Service Plan** (the compute), then the **Web App** (your application), and finally **deploy your code** into it — inside your existing resource group.

Define your variables (only the first line to edit — use **your first name**, lowercase, no accent):

  ```bash
  # macOS / Linux / Git Bash
  FIRSTNAME=frederic            # ← the ONLY line to edit
  RG=$FIRSTNAME                 # your resource group (already created for you)
  PLAN=${FIRSTNAME}plan
  API_NAME=${FIRSTNAME}wineapi  # must be globally unique: becomes <API_NAME>.azurewebsites.net
  LOCATION=swedencentral
  ```

  ```powershell
  # Windows (PowerShell)
  $FIRSTNAME="frederic"         # ← the ONLY line to edit
  $RG=$FIRSTNAME
  $PLAN="${FIRSTNAME}plan"
  $API_NAME="${FIRSTNAME}wineapi"
  $LOCATION="swedencentral"
  ```

#### 6.1 Create the App Service Plan

  ```bash
  az appservice plan create \
      -n $PLAN \
      -g $RG \
      -l $LOCATION \
      --is-linux \
      --sku B1
  ```

  🧠 **What does this do?** The *App Service Plan* is the **compute**: the underlying machine (Linux), its size and price (`B1` = Basic, 1 core / 1.75 GB RAM). Several Web Apps can share one plan — they'd share the same machine.

  🛠️ **If this fails** with a *quota* or *no available instances* error: don't retry in a loop — raise your hand. The shared subscription has limited App Service capacity, and **we have already pre-created a plan for you** in your resource group as a backup. A TA will point you to it (you'll just use that plan's name in the next step).

  👀 **Go check in the portal**: open [portal.azure.com](https://portal.azure.com), find your resource group, and look at your App Service Plan — its *Pricing plan* (B1) and *Operating System* (Linux).

#### 6.2 Create the Web App

  ```bash
  az webapp create \
      -n $API_NAME \
      -g $RG \
      -p $PLAN \
      --runtime "PYTHON:3.11"
  ```

  🧠 **What does `create` do?** It creates the **Web App** itself — an empty application slot running on your plan. Think of it as building the house (the plan is the land, the web app is the house) — but it has **no code inside yet**. `--runtime "PYTHON:3.11"` tells Azure to prepare a Python 3.11 environment.

  👀 **Go check in the portal**: open your Web App, find its **Default domain** (`https://<yourapi>.azurewebsites.net`) and open it — you'll see Azure's default landing page. The app exists, but *your* code isn't there yet.

#### 6.3 Configure how Azure builds and starts your app

Before sending your code, tell Azure two things:

  ```bash
  # 1. Install your requirements.txt when you deploy code
  az webapp config appsettings set \
      -g $RG \
      -n $API_NAME \
      --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true

  # 2. The command that launches your API server
  az webapp config set \
      -g $RG \
      -n $API_NAME \
      --startup-file "gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app"
  ```

  🧠 **What does this do?** The first command enables the build step at deployment time: Azure runs `pip install -r requirements.txt` for you on the server. The second sets the *Startup Command* — `gunicorn` with uvicorn workers, the production server that launches your FastAPI app (`main:app` = the `app` object in `main.py`).

#### 6.4 Deploy your code

Now send your code into the Web App. From your project root:

  ```bash
  cd # Gets back to the $HOME directory
  cd Documents/GitHub/yourwineapi

  az webapp up \
      -n $API_NAME \
      -g $RG \
      -p $PLAN \
      --runtime "PYTHON:3.11"
  ```

  🧠 **What does `up` do?** Where `create` made an empty app, `up` **packages the content of the current folder** (that's why the `cd` matters!) and uploads it. Azure then unpacks it, runs the build (installs your `requirements.txt`), and (re)starts the app with your Startup Command. In short: **`create` builds the house, `up` moves your furniture in.**

  The first deployment can take a few minutes — be patient. ☕

  Well done! You have deployed your first API to Azure. 🎉


---
### 7. Explore the portal and understand what you built

Before testing, take 10 minutes to **explore your resources in the portal** — this is how you'll debug real deployments:

- Open your **Web App** in the portal. Find:
  - the **Default domain** (`https://<yourapi>.azurewebsites.net`),
  - **Settings → Configuration**: spot your `SCM_DO_BUILD_DURING_DEPLOYMENT` setting and your Startup Command. Everything you did with the CLI is visible (and editable) here — the CLI and the portal are two interfaces to the same resources.
  - **Deployment Center → Logs** and **Log stream** (under Monitoring): this is where you watch the build and see your server starting. **Reading these logs is THE reflex to debug a deployment.**

💬 **Take a moment to ask questions**: What is the difference between the plan and the web app? What would change if you picked a bigger SKU? Where does the cost come from? Discuss with a TA.

---
### 8. Test your API

- Open your **Default domain** url in a new tab. You should reach the root endpoint. Add `/docs` to test the `/predict` endpoint, exactly as you did locally.

- Open `test.py` in VS Code, change the URL to your Web App domain, run it and check you get predictions.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 211814.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNDJIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--ed9c79dc73c59e1ff54c08ff956b3cbcd57e22a0/Capture%20d'%C3%A9cran%202024-09-22%20211814.png?disposition=attachment" />
</figure>

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 211846.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNDZIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--4510164d2c2456f900838fc9c9b2e50f9f0c2e24/Capture%20d'%C3%A9cran%202024-09-22%20211846.png?disposition=attachment" />
</figure>

🛠️🆘 **Troubleshooting**: if the app doesn't respond, check the **Log stream** in the portal, or restart:

  ```bash
  az webapp restart -n $API_NAME -g $RG
  ```

Be patient — a deployment can take several minutes to come up.




---
### 9. (Bonus) Automate your deployment with CI/CD

Right now, each change means re-running `az webapp up`. In a real MLOps workflow this is automated: **every push to GitHub runs the tests and, if they pass, deploys the new version**. This is CI/CD.

- Connect your Web App to GitHub from the portal: open your **Web App → Deployment Center** → Source: **GitHub** → authorize → select your organization, the `yourwineapi` repository and the `master` branch → **Save**.

<figure style="width: 50%">
  <img alt="Azure Deployment Center" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBMmtOQnc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--1b60f181919cc93883e728a3bdf0a0cac40b6955/Capture%20d%E2%80%99e%CC%81cran%202026-06-14%20a%CC%80%2019.47.01.png" />
</figure>

This file will allow you to deploy the API with a CI/CD approach. Every time you make a change locally, GitHub Actions will deploy the application in two steps: 
  * **build** the artifacts and make the tests in a virtual machine.
  * if the build step finished successfully, then, github will **deploy** the app to Azure. 

---
### 7. Finish the Continuous Integration with the tests.

- Go back to GitHub Desktop and use `git pull` to get the .yml file from the remote repository.

- Open the .yml file with VS Code and modify the build step. Find the comment to add the step to run tests (after installing dependencies). Then, add the last 2 lines to perform the tests. 

``` 
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      # Optional: Add step to run tests here (PyTest, Django test suites, etc.)
      - name: Test with pytest
        run: pytest tests/test_fast.py
```

- Save the File and commit the project. Then, push the project to GitHub and check the GitHub Actions. You will observe that the steps run correctly and update the Web App in Azure.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 205956.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNHVIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--132fb50a04a8cc4b3f768354a9525f7ba6982b64/Capture%20d'%C3%A9cran%202024-09-22%20205956.png?disposition=attachment" />
</figure>


<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 212449.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNHlIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--df9090514ef287e6233dcc7a6eb6cc75ca03ef6f/Capture%20d'%C3%A9cran%202024-09-22%20212449.png?disposition=attachment" />
</figure>


---
### 8. Test your API

- Go to your Web App panel and look for the `Default Domain`. Copy this url and open it in a new browser tab. You should be able to request the root endpoint of the API. If you add `/docs` to the url, you will be able to test the `/predict` endpoint as you did it locally. 

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 211814.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNDJIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--ed9c79dc73c59e1ff54c08ff956b3cbcd57e22a0/Capture%20d'%C3%A9cran%202024-09-22%20211814.png?disposition=attachment" />
</figure>


<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 211846.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNDZIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--4510164d2c2456f900838fc9c9b2e50f9f0c2e24/Capture%20d'%C3%A9cran%202024-09-22%20211846.png?disposition=attachment" />
</figure>


#### 🛠️🆘 Troubleshooting: if you have problems with your app, you might need to restart the machine in the Web App portal in Azure or redeploy the steps in GitHub Action. Just be patient, it can time several minutes to work.

- Open the `test.py` file with VS Code and modify the URL with your new Web app domain. Try the code and check that you can make predictions. 

Well done! You have deployed your first API to Azure with an MLOps approach. 

---

### Don't forget to save your work!

Save your files: File > Save. Then close all the tabs in your browser and VS Code windows. You can safely close the Anaconda Prompt.

💡 Don't forget to **push your code to GitHub**

1. Open GitHub Desktop.
2. It should automatically detect any file with modifications. If not, ask a TA.
3. Make sure these files are ticked, and write a _commit message_ in the bottom left form.
4. Click on the "Commit to `master`" button at the bottom of the form.
5. Click on the "Push `origin`" button at the top of the window.

---
## 🥇 Key learning points
By the end of this exercise, you will have:
* Understood how Azure is organized (Subscription → Resource Group → Resources) and worked inside your own resource group.
* Created the core Azure resources (App Service Plan, Web App) with the Azure CLI.
* Deployed a FastAPI API to Azure Web App Services.
* (Bonus) Automated tests and deployment with GitHub Actions following a CI/CD approach.

That's it! Take a small break before diving into the next exercise.