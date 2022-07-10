import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import copy
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import torchvision.models as models
from torch.autograd import Variable
from vision_transformer_pytorch import VisionTransformer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt


#positions are here learned not fixedly added
#instead of linear leaye, conv2d can be added
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 512, patch_size: int = 4, emb_size: int = 768, img_size: int = 28):#in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224
        self.patch_size = patch_size
        super().__init__()
        """
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        """
        #print("PATCH EMBEDDING...")
        self.patch_size = patch_size
        self.linear = nn.Linear(patch_size * patch_size * in_channels, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1,emb_size))
                
    def forward(self, x: Tensor):
        b,c,h,w = x.shape
        #x = self.projection(x)
        x = rearrange(x,"b c (h s1) (w s2) -> b (h w) (s1 s2 c)", s1=self.patch_size, s2=self.patch_size)
        x = self.linear(x)
        cls_tokens = repeat(self.cls_token,'() n e -> b n e', b=b)#repeat the cls tokens for all patch set in 
        x = torch.cat([cls_tokens,x],dim=1)
        x+=self.positions
        return x

class multiHeadAttention(nn.Module):
  def __init__(self, emb_size: int=768, heads: int=8, dropout: float=0.0):
    super().__init__()
    self.heads = heads
    self.emb_size = emb_size
    self.query = nn.Linear(emb_size,emb_size)
    self.key = nn.Linear(emb_size,emb_size)
    self.value = nn.Linear(emb_size,emb_size)
    self.drop_out = nn.Dropout(dropout)
    self.projection = nn.Linear(emb_size,emb_size)

  def forward(self,x):
    #splitting the single input int number of heads
    queries = rearrange(self.query(x),"b n (h d) -> b h n d", h = self.heads)
    keys = rearrange(self.key(x),"b n (h d) -> b h n d", h = self.heads)
    values = rearrange(self.value(x),"b n (h d) -> b h n d", h = self.heads)

    attention_maps = torch.einsum("bhqd, bhkd -> bhqk",queries,keys)
    scaling_value = self.emb_size**(1/2)
    attention_maps = F.softmax(attention_maps,dim=-1)/scaling_value
    attention_maps = self.drop_out(attention_maps)##might be deleted

    output = torch.einsum("bhal, bhlv -> bhav",attention_maps,values)
    output  = rearrange(output,"b h n d -> b n (h d)")
    output = self.projection(output)
    return output

class residual(nn.Module):
  def __init__(self,fn):
    super().__init__()
    self.fn = fn
  def forward(self,x):
    identity = x
    res = self.fn(x)
    out = res + identity
    return out

class mlp(nn.Sequential):#multi layer perceptron
  def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerBlock(nn.Sequential):
  def __init__(self,emb_size:int = 768,drop_out:float=0.0):
    super().__init__(
        residual(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                multiHeadAttention(emb_size),
                nn.Dropout(drop_out)
            )
        ),
        residual(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                mlp(emb_size),
                nn.Dropout(drop_out)
            )
        )
    )

class Transformer(nn.Sequential):
  def __init__(self,loops:int =12):
    super().__init__(
        *[TransformerBlock() for _ in range(loops)]
    )

class Classification(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)
    def forward(self, x: Tensor):
        x = reduce(x,'b n e -> b e', reduction='mean')
        x = self.norm(x)
        output = self.linear(x)
        return output
        
class Classification_old(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

class VIT(nn.Module):
  def __init__(self,resnetM,emb_size: int=768,drop_out: float=0.0, n_classes:int = 3,in_channels:int=512,patch_size:int=4,image_size:int=28):
    super().__init__()
    print("INIT VIT")
    self.resnetM = resnetM
    self.PatchEmbedding = PatchEmbedding(in_channels,patch_size,emb_size,image_size)
    self.Transformer = Transformer()
    self.Classification = Classification(n_classes=3)
  def forward(self,x):
    resnetOutput = self.resnetM(x)
    # print("FORWARD VIT")
    #print(resnetOutput.shape)
    patchEmbeddings = self.PatchEmbedding(resnetOutput)
    transformerOutput = self.Transformer(patchEmbeddings)
    classificationOutput = self.Classification(transformerOutput)
    #print(classificationOutput)
    output = F.log_softmax(classificationOutput, dim=1)
    return output
    #return classificationOutput
################################

def trainAttentionModel(model, criterion, optimizer, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:#, 'val'
            """
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            """
            model.train()  # Set model to training mode
            current_loss = 0.0
            current_corrects = 0

            # Here's where the training happens
            print('Iterating through data...')

            for inputs, labels in train:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # We need to zero the gradients, don't forget it
                optimizer.zero_grad()

                # Time to carry out the forward training poss
                # We only need to log the loss stats if we are in training phase
                with torch.set_grad_enabled(phase == 'train'):#phase == 'train'
                    outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    #print(labels)
                    #print("############", outputs,"############", preds)
                    #print("LABELS SHAPE: ", labels.shape)
                    #print("OUTPUTS SHAPE: ", outputs.shape)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    #if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # We want variables to hold the loss statistics
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / train_size#dataset_sizes[phase]
            epoch_acc = current_corrects.double() / train_size#dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Make a copy of the model if the accuracy on the validation set has improved
            if epoch_acc > best_acc:#phase == 'val' and 
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    print('Finished Training')

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    return model


##################
# from ResnetVit_brain_test_prediction import VIT
# def getVitClassify():
print(torch.cuda.get_device_name(1))
print(torch.cuda.is_available())
if torch.cuda.is_available():  
    dev = "cuda:1" 
else:  
    dev = "cpu"
print("Number of GPU: ", torch.cuda.device_count())
device = torch.device(dev) 
print(device)
device = torch.device('cuda:1')#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Name of current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Memory Summary before: ", torch.cuda.memory_summary())
torch.cuda.empty_cache()
print("Memory Summary after empty_cache: ", torch.cuda.memory_summary())

image_transforms = transforms.Compose([
                                transforms.Resize((384,384)),#224,224
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))#(0.5, 0.5, 0.5), (0.5, 0.5, 0.5) 
])

# data_dir = '/home/user/Documents/webapp/static/vit'
data_dir = '/home/user/Documents/Db-Stroke-Split/Dataloader-train-test-65-35/'

train_set = ImageFolder(data_dir+"Train_Set", transform=image_transforms)
test_set = ImageFolder(data_dir+"Test_Set", transform=image_transforms)

print(test_set.classes)
print(test_set.class_to_idx)
train = DataLoader(train_set, batch_size=1, shuffle=True, drop_last=True)
test = DataLoader(test_set, batch_size=1, shuffle=True, drop_last=True)

#print("Train images", len(train))
print("Test images", len(test), test, type(test))

# for idx, batch in enumerate(test):
#     print(idx, " ", batch)
#     print(idx.classes)
#dataset = ImageFolder('/home/user/Documents/Db-Stroke-full', transform=image_transforms)
train_size = len(train)
# batch_size=2#len(dataset)
#dataset = DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)#True

preTrainedModel = models.resnet50(pretrained=True)
preTrainedModel = torch.nn.Sequential(*(list(preTrainedModel.children())[:-4]))#B*512*28*28
#print(preTrainedModel,preTrainedModel(img).shape)

Vnet = VisionTransformer.from_pretrained('R50+ViT-B_16')
resnetM = torch.nn.Sequential(*(list(Vnet.children())[:-3]))
print("in here 1")
model = VIT(resnetM,768,0.0,3,1024,1,24)
print("in here 2")
model.to(device)
print("after device")
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# model = trainAttentionModel(model,criterion,optimizer_ft,10)
# torch.save(model,"Resnet_VIT_Updated_New_F1.pth")
model = torch.load("Resnet_VIT_Updated_New_F1_for_graph_patch_1.pth")
# model.load_state_dict(torch.load("Resnet_VIT_Updated_New_F1_Dics.pth"))
print("MODEL LOADED", "="*50)
correct = 0
total = 0
act = []
pre = []
with torch.no_grad():
    for data in test:
        images, labels = data
        #print(data)
        images = images.to(device)
        labels = labels.to(device)
        #print(data.classes)
        print("READING IMAGES")
        outputs = model(images)
        print("PREDICTING.....")
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        #print("Predicted: ", predicted, "Labels: ", labels)
        correct += (predicted == labels).sum().item()
        # print(labels.item())
        # print("**",predicted.item())
        act.append(labels.item())
        pre.append(predicted.item())

print('Accuracy of the network on the %d test images: %d %%' % (len(test),
    100 * correct / total))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(act, pre))
target_names = ['Hemm', 'Infa', 'Norm']
print(act, pre)
print(classification_report(act, pre, target_names=target_names))
print("DONE **********************************")
# getVitClassify()