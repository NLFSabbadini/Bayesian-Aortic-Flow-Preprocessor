from vmtk import vmtkscripts


"""Instantiate vmtkScript and set its variables in a single command"""
def build(name, **args):
	name = f"vmtk{name}"
	if name not in vmtkscripts.__dict__.keys():
		raise AttributeError(f"module \'vmtk.vmtkscripts\' has no attribute \'{name}\'")
	instance = vmtkscripts.__dict__[name]()
	for key, val in args.items():
		if key not in instance.__dict__.keys():
			raise AttributeError(f"\'{name}\' object has no attribute \'{key}\'")
		instance.__dict__[key] = val
	return instance


"""Instantiate vmtkScript and execute"""
def run(name, **args):
	instance = build(name, **args)
	instance.Execute()
	return instance
