class DQN(nn.Module):
	def __init__(self, img_height, img_width):
		super().__init__()

		self.fc1 = nn.Linear(in_fatures=img_height*img_width*3, out_features = 24)
		self.fc2 = nn.Linear(in_fatures=24, out_features = 32)
		self.out = nn.Linear(in_fatures=32, out_features = 2)

	def forward(self, t):
		t = t.flatten(start_dim=1)
		t = F.relu(self.fc1(t))
		t = F.relu(self.fc1(t))
		t = self.out(t)
		return t