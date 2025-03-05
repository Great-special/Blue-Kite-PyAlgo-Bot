import flet as ft


def main(page: ft.Page):
    page.title = "To-Do List"
    page.theme_mode = ft.ThemeMode.SYSTEM
    
    
    task_input = ft.TextField(label='Add a task', width=400, autofocus=True)
    add_task_button = ft.ElevatedButton(text='Add Task', icon=ft.icons.ADD, on_click=lambda e: add_task())
    task_list = ft.ListView(expand=True, spacing=10)
    theme_button = ft.ElevatedButton("Change Theme Mode", icon=ft.icons.SETTINGS, on_click=lambda e: change_theme())
    
    page.appbar = ft.AppBar(title=ft.Text("To Do"), bgcolor="red", center_title=True, actions=[ft.Icon(name="settings")])
    page.navigation_bar = ft.NavigationBar(destinations=[
        ft.NavigationBarDestination(label="Explore", icon=ft.icons.EXPLORE),
        ft.NavigationBarDestination(label='Commute', icon=ft.icons.COMMUTE),
        ft.NavigationBarDestination(label='Home', icon=ft.icons.HOME)
    ])
    
    
    def add_task():
        if task_input.value.strip():
            task_item = ft.Container(
                content=ft.Text(value=task_input.value),
                padding=10,
                border_radius=5,
                bgcolor='lightblue'
                ),

        task_list.controls.append(task_item)
        task_input.value = ''
        page.update()
    
    
    def change_theme():
        if page.theme_mode == ft.ThemeMode.DARK:
            page.theme_mode = ft.ThemeMode.LIGHT
        elif page.theme_mode == ft.ThemeMode.LIGHT:
            page.theme_mode = ft.ThemeMode.DARK
        else:
            page.theme_mode = ft.ThemeMode.LIGHT
        
        page.update()
            
    def complete_task(e):
        e.control.content.color = 'gray'
        e.control.bgcolor = 'lightgray'
        e.control.update()        
    
    
    page.add(
        ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        task_input,
                        add_task_button
                    ],
                ),
                theme_button,
                task_list
            ]
        )
    )
    
    
    


ft.app(main)
