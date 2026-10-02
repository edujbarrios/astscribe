from astscribe import explain

code = """
model.eval()
with torch.no_grad():
    outputs = model(inputs)
"""

print(explain(code, style="scientific"))
