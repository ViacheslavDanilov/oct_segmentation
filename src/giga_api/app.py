import base64
import os
from io import BytesIO

from dotenv import load_dotenv
from gigachat import GigaChat
from PIL import Image

load_dotenv()

GIGA_CHAT_TOKEN = os.getenv("GIGA_CHAT_TOKEN")


class GigaChatModel:
    """Класс для работы с моделью GigaChat."""

    def __init__(self, auth_token):
        self.auth_token = auth_token
        self.giga = GigaChat(
            credentials=auth_token,
            model="GigaChat-2-Pro",
            verify_ssl_certs=False,
        )
        response = self.giga.get_token()
        self.access_token = response.access_token

    def get_img_chat_message(self, prompt: str, image: Image.Image = None):
        # image = self.get_pil_to_base64(image)

        # image.save('temp.jpg')
        # file = self.giga.upload_file(image.tobytes())
        # file = self.giga.upload_file(open('temp.jpg', 'rb'))
        # print(file)

        if image is not None:
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="JPEG")
            img_byte_arr.seek(0)

        # headers = {
        #     'Authorization': f'Bearer {self.access_token}'
        # }
        #
        # files = {
        #     "file": ("image.jpeg", img_byte_arr, "image/jpeg"),
        #     "purpose": (None, "general")
        # }
        #
        # response = requests.request(
        #     "POST",
        #     f"https://gigachat.devices.sberbank.ru/api/v1/files",
        #     headers=headers,
        #     files=files,
        #     verify=False
        # )
        # response = response.json()

        # print(response.json())

        response = self.giga.chat(
            {
                "function_call": "auto",
                "messages": [
                    {
                        "role": "system",
                        "content": "You work as a medical imaging and biometric data analysis system, specifically designed for forming conclusions based on Optical Coherence Tomography (OCT) results. Your task is to identify potential risks of atherosclerotic plaque rupture by evaluating specific indicators, including the average area of the vascular lumen, the number, thickness, and characteristics of fibrous caps. Your goal is to provide clinicians with clear and objective conclusions regarding the patient's vascular condition and the risk of cardiovascular complications. No need to use filler words, only clear and understandable answers. Also, do not add reasoning that there is insufficient data, cannot give an accurate answer, etc. Write medical facts and observations. Base your response on this example: The average vascular lumen area is 30 ± 4 μm², which may indicate the presence of stenosis. The fibrous cap has a minimum thickness of only 0.19 μm, which is a very low value and indicates a high risk of atherosclerotic plaque rupture. The average thickness of the fibrous cap is 0.79 ± 0.21 μm, with high variability in values (standard deviation 0.21 μm). The total number of objects (presumably atherosclerotic plaques) is five, which may also indicate the prevalence of the process. The OCT data indicate an increased risk of atherosclerotic plaque rupture due to a thin and unstable fibrous cap. A consultation with a cardiologist or vascular surgeon is recommended to determine further patient management tactics, which may include drug therapy, lifestyle changes, and possibly invasive interventions. Answer strictly in JSON format with fields: description: str(description of the medical conclusion), risk_classification: str(low, medium, high)",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0.64,
            },
        )
        return response

    @staticmethod
    def get_pil_to_base64(image: Image.Image) -> str:
        return base64.b64encode(image.tobytes()).decode("utf-8")


gigachat = GigaChatModel(
    auth_token=GIGA_CHAT_TOKEN,
)
